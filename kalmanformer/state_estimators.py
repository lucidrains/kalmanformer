import torch
from torch import einsum
from torch.nn import Module

from einops import repeat

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def to_callable(obj):
    if not exists(obj):
        return None
    if callable(obj):
        return obj
    def _callable(step):
        return obj[:, step] if obj.ndim == 4 else obj
    return _callable

# base class - standardized interface for state estimators
# mems carries whatever internal state the estimator needs between steps

class BaseStateEstimator(Module):

    def step(self, z_k, z_prev, x_prev_post, x_prev_prior, F_k, H_k, mems = None, step = None):
        raise NotImplementedError

    def forward(
        self,
        observations,       # (b, seq, obs_dim)
        F,                  # callable or (state_dim, state_dim) or (b, seq, state_dim, state_dim)
        H,                  # callable or (obs_dim, state_dim) or (b, seq, obs_dim, state_dim)
        x_0 = None,         # (b, state_dim)
        initial_state = None,
        return_state = False
    ):
        b, seq_len, _, device = *observations.shape, observations.device

        F_fn = to_callable(F)
        H_fn = to_callable(H)

        obs_seq = observations.unbind(dim = 1)

        if exists(initial_state):
            # resume from cached state
            # caller passes F/H aligned with observations

            x_prev_post, x_prev_prior, z_prev, mems = initial_state
            post_states = []

            iter_obs = obs_seq
            start_seq_idx = 0
        else:
            # fresh start - first observation used as z_prev

            x_prev_post = default(x_0, torch.zeros(b, self.state_dim, device = device))
            x_prev_prior = x_prev_post.clone()
            z_prev = obs_seq[0]
            mems = None
            post_states = [x_prev_post]

            iter_obs = obs_seq[1:]
            start_seq_idx = 1

        for step, z_k in enumerate(iter_obs):
            seq_idx = start_seq_idx + step

            F_idx = seq_idx - 1 if not exists(initial_state) else seq_idx
            H_idx = seq_idx

            F_k = F_fn(F_idx) if exists(F_fn) else None
            H_k = H_fn(H_idx) if exists(H_fn) else None

            x_prev_post, x_prev_prior, _, mems = self.step(
                z_k, z_prev, x_prev_post, x_prev_prior, F_k, H_k, mems = mems, step = step
            )
            post_states.append(x_prev_post)
            z_prev = z_k

        out = torch.stack(post_states, dim = 1)

        if not return_state:
            return out

        return out, (x_prev_post, x_prev_prior, z_prev, mems)

# extended kalman filter as a module

class ExtendedKalmanFilter(BaseStateEstimator):
    def __init__(
        self,
        *,
        state_dim,
        obs_dim,
        Q = None,
        R = None
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.register_buffer('Q', default(Q, torch.eye(state_dim)))
        self.register_buffer('R', default(R, torch.eye(obs_dim)))
        self.register_buffer('I', torch.eye(state_dim), persistent = False)

    def step(
        self,
        z_k,
        z_prev,
        x_prev_post,
        x_prev_prior,
        F_k,
        H_k,
        mems = None,
        step = None
    ):
        b = z_k.shape[0]

        # mems carries the posterior covariance P

        P_post = default(mems, repeat(torch.zeros_like(self.Q), 'i j -> b i j', b = b))

        # predict

        x_prior = einsum('... i j, ... j -> ... i', F_k, x_prev_post)

        P_prior = einsum('... i j, ... j k -> ... i k', F_k, P_post)
        P_prior = einsum('... i j, ... k j -> ... i k', P_prior, F_k) + self.Q

        # innovation

        z_prior = einsum('... i j, ... j -> ... i', H_k, x_prior)
        innovation = z_k - z_prior

        # kalman gain

        S = einsum('... i j, ... j k -> ... i k', H_k, P_prior)
        S = einsum('... i j, ... k j -> ... i k', S, H_k) + self.R

        K_k = einsum('... i j, ... k j -> ... i k', P_prior, H_k)
        K_k = einsum('... i j, ... j k -> ... i k', K_k, torch.inverse(S))

        # update

        x_post = x_prior + einsum('... i j, ... j -> ... i', K_k, innovation)

        I = repeat(self.I, 'i j -> b i j', b = b)
        KH = einsum('... i j, ... j k -> ... i k', K_k, H_k)
        next_P = einsum('... i j, ... j k -> ... i k', I - KH, P_prior)

        return x_post, x_prior, K_k, next_P
