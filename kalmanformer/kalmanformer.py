import torch
from torch import nn, einsum, stack
from torch.nn import Module

from einops import rearrange

from x_transformers import Encoder
from x_mlps_pytorch import create_mlp

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main class

class KalmanFormer(Module):
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
        mlp_dim = 64,
        enc_kwargs: dict = dict(),
        dec_kwargs: dict = dict(),
        learn_dynamics: bool = False,
        use_memory: bool = False
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.learn_dynamics = learn_dynamics
        self.use_memory = use_memory

        if learn_dynamics:
            self.learned_F = create_mlp(
                dim = dim,
                depth = 2,
                dim_in = state_dim,
                dim_out = state_dim
            )
            self.learned_H = nn.Linear(state_dim, obs_dim, bias = False)

        self.proj_obs_diff = nn.Linear(obs_dim, dim)
        self.proj_innov_diff = nn.Linear(obs_dim, dim)

        self.proj_state_evol_diff = nn.Linear(state_dim, dim)
        self.proj_state_upd_diff = nn.Linear(state_dim, dim)

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

        self.to_kalman_gain = create_mlp(
            dim = mlp_dim,
            depth = mlp_depth,
            dim_in = dim * 2,
            dim_out = state_dim * obs_dim
        )

        nn.init.zeros_(self.to_kalman_gain.layers[-1].weight)
        nn.init.zeros_(self.to_kalman_gain.layers[-1].bias)

    def step(
        self,
        z_k,            # (b, obs_dim)
        z_prev,         # (b, obs_dim)
        x_prev_post,    # (b, state_dim)
        x_prev_prior,   # (b, state_dim)
        F_k,            # (state_dim, state_dim) or (b, state_dim, state_dim)
        H_k,            # (obs_dim, state_dim) or (b, obs_dim, state_dim)
        mems = None
    ):
        x_prior = einsum('... i j, ... j -> ... i', F_k, x_prev_post) if exists(F_k) else self.learned_F(x_prev_post)
        z_prior = einsum('... i j, ... j -> ... i', H_k, x_prior) if exists(H_k) else self.learned_H(x_prior)

        obs_diff = z_k - z_prev
        innov_diff = z_k - z_prior
        state_evol_diff = x_prior - x_prev_post
        state_upd_diff = x_prev_post - x_prev_prior

        token_obs_diff = self.proj_obs_diff(obs_diff)
        token_innov_diff = self.proj_innov_diff(innov_diff)

        token_state_evol_diff = self.proj_state_evol_diff(state_evol_diff)
        token_state_upd_diff = self.proj_state_upd_diff(state_upd_diff)

        enc_in = stack((token_obs_diff, token_innov_diff), dim = 1)
        enc_out = self.obs_encoder(enc_in)

        dec_in = stack((token_state_evol_diff, token_state_upd_diff), dim = 1)

        if self.use_memory:
            dec_out, intermediates = self.state_encoder(dec_in, context = enc_out, mems = mems, return_hiddens = True)
            next_mems = intermediates.hiddens
        else:
            dec_out = self.state_encoder(dec_in, context = enc_out)
            next_mems = None

        dec_flat = rearrange(dec_out, 'b n d -> b (n d)')
        K_k = rearrange(self.to_kalman_gain(dec_flat), 'b (i j) -> b i j', i = self.state_dim, j = self.obs_dim)

        update = einsum('b i j, b j -> b i', K_k, innov_diff)
        x_post = x_prior + update

        return x_post, x_prior, K_k, next_mems

    def forward(
        self,
        observations,   # (b, seq, obs_dim)
        F,              # (state_dim, state_dim) or (b, seq, state_dim, state_dim)
        H,              # (obs_dim, state_dim) or (b, seq, obs_dim, state_dim)
        x_0 = None      # (b, state_dim)
    ):
        b, device = observations.shape[0], observations.device

        x_prev_post = default(x_0, torch.zeros(b, self.state_dim, device = device))
        x_prev_prior = x_prev_post.clone()

        observations = observations.unbind(dim = 1)
        z_prev = observations[0]

        post_states = [x_prev_post]
        mems = None

        for k, z_k in enumerate(observations[1:], start = 1):

            F_k = F[:, k - 1] if exists(F) and F.ndim == 4 else F
            H_k = H[:, k] if exists(H) and H.ndim == 4 else H

            x_prev_post, x_prev_prior, K_k, mems = self.step(
                z_k, z_prev, x_prev_post, x_prev_prior, F_k, H_k,
                mems = mems
            )

            post_states.append(x_prev_post)
            z_prev = z_k

        return stack(post_states, dim = 1)
