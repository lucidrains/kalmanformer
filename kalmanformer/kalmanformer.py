import torch
from torch import nn, einsum, stack

from einops import rearrange
from einops.layers.torch import Rearrange

from x_transformers import Encoder
from x_mlps_pytorch import create_mlp
from torch_einops_utils import tree_map_tensor

from kalmanformer.state_estimators import BaseStateEstimator, exists, default

# helpers

def divisible_by(num, den):
    return (num % den) == 0

def detach_mems(mems):
    return tree_map_tensor(lambda t: t.detach(), mems)

def init_zero_(layer):
    nn.init.zeros_(layer.weight)
    if exists(layer.bias):
        nn.init.zeros_(layer.bias)

# kalmanformer

class KalmanFormer(BaseStateEstimator):
    def __init__(
        self,
        *,
        state_dim,
        obs_dim,
        dim = 64,
        depth = 2,
        heads = 2,
        dim_head = 32,
        ff_mult = 1,
        mlp_depth = 2,
        mlp_dim = None,
        enc_kwargs: dict | None = None,
        dec_kwargs: dict | None = None,
        learn_dynamics = False,
        use_memory = False,
        base_estimator: BaseStateEstimator | None = None,
        detach_mems_every = 2
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.learn_dynamics = learn_dynamics
        self.use_memory = use_memory
        self.base_estimator = base_estimator

        self.detach_mems_every = detach_mems_every
        self.should_detach_mems = detach_mems_every > 0

        mlp_dim = default(mlp_dim, dim)
        enc_kwargs = default(enc_kwargs, dict())
        dec_kwargs = default(dec_kwargs, dict())

        has_base = exists(base_estimator)

        # learned dynamics via GRU

        if learn_dynamics:
            self.dynamics_rnn = nn.GRU(
                input_size = state_dim,
                hidden_size = dim,
                num_layers = 2,
                batch_first = True
            )

            self.to_F = nn.Sequential(
                nn.Linear(dim, state_dim * state_dim),
                Rearrange('b (i j) -> b i j', i = state_dim, j = state_dim)
            )

            self.to_H = nn.Sequential(
                nn.Linear(dim, obs_dim * state_dim),
                Rearrange('b (i j) -> b i j', i = obs_dim, j = state_dim)
            )

            init_zero_(self.to_F[0])
            init_zero_(self.to_H[0])

            self.register_buffer('F_identity', torch.eye(state_dim), persistent = False)

            H_identity = torch.zeros(obs_dim, state_dim)
            min_dim = min(obs_dim, state_dim)
            H_identity[:min_dim, :min_dim] = torch.eye(min_dim)

            self.register_buffer('H_identity', H_identity, persistent = False)

        # observation encoder tokens

        self.proj_obs_diff = nn.Linear(obs_dim, dim)
        self.proj_innov_diff = nn.Linear(obs_dim, dim)

        # state decoder tokens

        self.proj_state_evol_diff = nn.Linear(state_dim, dim)
        self.proj_state_upd_diff = nn.Linear(state_dim, dim)

        num_state_tokens = 2

        if has_base:
            self.proj_base_update = nn.Linear(state_dim, dim)
            num_state_tokens += 1

        # encoder / decoder

        self.obs_encoder = Encoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head,
            ff_mult = ff_mult,
            verbose = False,
            **enc_kwargs
        )

        self.state_encoder = Encoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head,
            cross_attend = True,
            ff_mult = ff_mult,
            verbose = False,
            **dec_kwargs
        )

        # to kalman gain

        self.to_kalman_gain = create_mlp(
            dim = mlp_dim,
            depth = mlp_depth,
            dim_in = dim * num_state_tokens,
            dim_out = state_dim * obs_dim
        )

        init_zero_(self.to_kalman_gain.layers[-1])

    def step(
        self,
        z_k,            # (b, obs_dim)
        z_prev,         # (b, obs_dim)
        x_prev_post,    # (b, state_dim)
        x_prev_prior,   # (b, state_dim)
        F_k,            # Callable[[int], Tensor] | (state_dim, state_dim) | (b, state_dim, state_dim)
        H_k,            # Callable[[int], Tensor] | (obs_dim, state_dim) | (b, obs_dim, state_dim)
        mems = None,
        step = None
    ):
        has_base = exists(self.base_estimator)

        should_detach = (
            self.should_detach_mems
            and exists(step)
            and step > 0
            and divisible_by(step, self.detach_mems_every)
        )

        if should_detach:
            mems = detach_mems(mems)

        # unpack mems

        if has_base:
            base_mems, transformer_mems = mems if exists(mems) else (None, None)
        else:
            base_mems, transformer_mems = None, mems

        obs_mems, state_mems, rnn_hiddens = transformer_mems if exists(transformer_mems) else (None, None, None)

        if not self.use_memory:
            obs_mems, state_mems = None, None

        # learned dynamics

        if self.learn_dynamics:
            rnn_in = rearrange(x_prev_post, 'b d -> b 1 d')
            rnn_out, rnn_hiddens = self.dynamics_rnn(rnn_in, rnn_hiddens)
            rnn_out = rearrange(rnn_out, 'b 1 d -> b d')

            F_k = self.to_F(rnn_out) + (F_k if exists(F_k) else self.F_identity)
            H_k = self.to_H(rnn_out) + (H_k if exists(H_k) else self.H_identity)

        # predict prior state

        if has_base:
            base_x_post, x_prior, _, base_mems = self.base_estimator.step(
                z_k, z_prev, x_prev_post, x_prev_prior, F_k, H_k, mems = base_mems
            )
        else:
            x_prior = einsum('... i j, ... j -> ... i', F_k, x_prev_post)

        z_prior = einsum('... i j, ... j -> ... i', H_k, x_prior)

        # compute difference tokens

        obs_diff = z_k - z_prev
        innov_diff = z_k - z_prior
        state_evol_diff = x_prior - x_prev_post
        state_upd_diff = x_prev_post - x_prev_prior

        # observation encoder

        enc_in = stack((
            self.proj_obs_diff(obs_diff),
            self.proj_innov_diff(innov_diff),
        ), dim = 1)

        if self.use_memory:
            enc_out, obs_intermediates = self.obs_encoder(enc_in, mems = obs_mems, return_hiddens = True)
            next_obs_mems = obs_intermediates.hiddens
        else:
            enc_out = self.obs_encoder(enc_in)
            next_obs_mems = None

        # state decoder

        state_tokens = [
            self.proj_state_evol_diff(state_evol_diff),
            self.proj_state_upd_diff(state_upd_diff),
        ]

        if has_base:
            base_update = base_x_post - x_prior
            state_tokens.append(self.proj_base_update(base_update))

        dec_in = stack(state_tokens, dim = 1)

        if self.use_memory:
            dec_out, state_intermediates = self.state_encoder(dec_in, context = enc_out, mems = state_mems, return_hiddens = True)
            next_state_mems = state_intermediates.hiddens
        else:
            dec_out = self.state_encoder(dec_in, context = enc_out)
            next_state_mems = None

        # predict kalman gain and apply update

        dec_flat = rearrange(dec_out, 'b n d -> b (n d)')

        K_k = rearrange(
            self.to_kalman_gain(dec_flat),
            'b (i j) -> b i j', i = self.state_dim, j = self.obs_dim
        )

        update = einsum('b i j, b j -> b i', K_k, innov_diff)

        # kalman update

        x_post = x_prior + update

        # next mems

        next_transformer_mems = (next_obs_mems, next_state_mems, rnn_hiddens)
        next_mems = (base_mems, next_transformer_mems) if has_base else next_transformer_mems

        return x_post, x_prior, K_k, next_mems
