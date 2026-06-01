from torch import nn, einsum, stack

from einops import rearrange

from x_transformers import Encoder
from x_mlps_pytorch import create_mlp
from torch_einops_utils import tree_map_tensor

from kalmanformer.state_estimators import BaseStateEstimator, exists, default

# helpers

def divisible_by(num, den):
    return (num % den) == 0

def detach_mems(mems):
    return tree_map_tensor(lambda t: t.detach(), mems)

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

        if learn_dynamics:
            self.learned_F = create_mlp(
                dim = dim,
                depth = 2,
                dim_in = state_dim,
                dim_out = state_dim
            )
            self.learned_H = nn.Linear(state_dim, obs_dim, bias = False)

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

        nn.init.zeros_(self.to_kalman_gain.layers[-1].weight)
        nn.init.zeros_(self.to_kalman_gain.layers[-1].bias)

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

        if self.use_memory:
            obs_mems, state_mems = transformer_mems if exists(transformer_mems) else (None, None)
        else:
            obs_mems, state_mems = None, None

        # predict prior state (from base estimator or dynamics model)

        if has_base:
            base_x_post, x_prior, _, base_mems = self.base_estimator.step(
                z_k, z_prev, x_prev_post, x_prev_prior, F_k, H_k, mems = base_mems
            )
        else:
            x_prior = einsum('... i j, ... j -> ... i', F_k, x_prev_post) if exists(F_k) else self.learned_F(x_prev_post)

        z_prior = einsum('... i j, ... j -> ... i', H_k, x_prior) if exists(H_k) else self.learned_H(x_prior)

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
            next_transformer_mems = (next_obs_mems, next_state_mems)
        else:
            dec_out = self.state_encoder(dec_in, context = enc_out)
            next_transformer_mems = None

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

        next_mems = (base_mems, next_transformer_mems) if has_base else next_transformer_mems

        return x_post, x_prior, K_k, next_mems
