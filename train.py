# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "numpy",
#     "scipy",
#     "einops",
#     "fire",
#     "wandb",
#     "x-transformers",
#     "x-mlps-pytorch",
#     "accelerate"
# ]
# ///

import numpy as np
from scipy.integrate import odeint

import torch
from torch import einsum, tensor
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.optim import Adam

from einops import repeat

import fire
import wandb
from accelerate import Accelerator

from kalmanformer import KalmanFormer

# lorenz attractor

def lorenz_deriv(state, t):
    x, y, z = state
    return [
        10.0 * (y - x),
        x * (28.0 - z) - y,
        x * y - (8.0 / 3.0) * z
    ]

# computes the state transition matrix F_k using Taylor expansion
# F_k = I + \sum_{j=1}^5 (A(x_k) \Delta t)^j / j!

def get_jacobian(state, dt, order = 5, pseudo_linear = True):
    x, y, z = state

    x_ = 0. if pseudo_linear else x

    A = np.array([
        [-10., 10., 0.],
        [28. - z, -1., -x_],
        [y, x_, -8. / 3.]
    ])

    A_dt = A * dt

    F = np.eye(3)
    term = np.eye(3)

    for j in range(1, order + 1):
        term = (term @ A_dt) / j
        F += term

    return F

def generate_dataset(num_trajectories = 100, T = 5., dt = 0.05, process_noise_std = 0.05, obs_noise_std = 0.05):
    t = np.arange(0, T, dt)
    dataset_states = []
    dataset_obs = []
    dataset_F = []

    for _ in range(num_trajectories):
        x0 = np.random.uniform(-10, 10, size=(3,))
        states = odeint(lorenz_deriv, x0, t)

        observations = states + np.random.normal(0, obs_noise_std, size=states.shape)
        F_seq = np.array([get_jacobian(state, dt) for state in states])

        dataset_states.append(states)
        dataset_obs.append(observations)
        dataset_F.append(F_seq)

    return (
        tensor(np.array(dataset_states), dtype = torch.float32),
        tensor(np.array(dataset_obs), dtype = torch.float32),
        tensor(np.array(dataset_F), dtype = torch.float32)
    )

# extended kalman filter baseline

def run_ekf(observations, F_seq, H, Q, R, x0):
    b, seq_len, _ = observations.shape
    device = observations.device
    state_dim = x0.shape[1]

    post_states = [x0]

    x_post = x0
    P_post = torch.zeros(b, state_dim, state_dim, device = device)

    for k in range(1, seq_len):
        z_k = observations[:, k]
        F_k = F_seq[:, k - 1]

        x_prior = einsum('b i j, b j -> b i', F_k, x_post)
        P_prior = einsum('b i j, b j k -> b i k', F_k, P_post)
        P_prior = einsum('b i j, b k j -> b i k', P_prior, F_k) + Q

        S = einsum('i j, b j k -> b i k', H, P_prior)
        S = einsum('b i j, k j -> b i k', S, H) + R

        S_inv = torch.inverse(S)
        K_k = einsum('b i j, k j -> b i k', P_prior, H)
        K_k = einsum('b i j, b j k -> b i k', K_k, S_inv)

        z_prior = einsum('i j, b j -> b i', H, x_prior)
        innovation = z_k - z_prior

        x_post = x_prior + einsum('b i j, b j -> b i', K_k, innovation)

        I = repeat(torch.eye(state_dim, device = device), 'i j -> b i j', b = b)
        KH = einsum('b i j, j k -> b i k', K_k, H)
        P_post = einsum('b i j, b j k -> b i k', I - KH, P_prior)

        post_states.append(x_post)

    return torch.stack(post_states, dim = 1)

# training script

def train(
    use_wandb: bool = False,
    epochs: int = 10,
    batch_size: int = 30,
    num_train_trajectories: int = 100,
    num_test_trajectories: int = 20,
    sequence_length: float = 5.,
    dt: float = 0.05,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    use_kalmanformer: bool = True,
    learn_dynamics: bool = False,
    use_memory: bool = False,
    cpu: bool = False
):
    # trains and evaluates KalmanFormer vs EKF on the Lorenz Attractor toy task

    accelerator = Accelerator(cpu = cpu)
    device = accelerator.device

    if use_wandb and accelerator.is_main_process:
        wandb.init(project = 'kalmanformer', config = locals())

    accelerator.print('generating dataset...', flush=True)

    train_states, train_obs, train_F = generate_dataset(num_trajectories = num_train_trajectories, T = sequence_length, dt = dt)
    test_states, test_obs, test_F = generate_dataset(num_trajectories = num_test_trajectories, T = sequence_length, dt = dt)

    train_states, train_obs, train_F = train_states.to(device), train_obs.to(device), train_F.to(device)
    test_states, test_obs, test_F = test_states.to(device), test_obs.to(device), test_F.to(device)

    H = torch.eye(3, device = device)

    # baseline EKF

    Q = torch.eye(3, device = device) * 0.8**2
    R = torch.eye(3, device = device) * 1.0**2

    accelerator.print('\nrunning ekf baseline...\n')
    ekf_test_preds = run_ekf(test_obs, test_F, H, Q, R, test_states[:, 0])
    ekf_mse = F.mse_loss(ekf_test_preds, test_states)
    accelerator.print(f'ekf test mse: {ekf_mse.item():.4f}')

    if use_wandb and accelerator.is_main_process:
        wandb.log({'test/ekf_mse': ekf_mse.item()})

    if not use_kalmanformer:
        return

    # kalmanformer

    accelerator.print(f'\ntraining kalmanformer (use_memory={use_memory})...\n')

    model = KalmanFormer(
        state_dim = 3,
        obs_dim = 3,
        dim = 64,
        depth = 2,
        heads = 2,
        learn_dynamics = learn_dynamics,
        use_memory = use_memory
    )

    optimizer = Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

    dataset = TensorDataset(train_states, train_obs, train_F)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for states, obs, F_seq in dataloader:

            preds = model(
                obs,
                F = None if learn_dynamics else F_seq,
                H = None if learn_dynamics else H,
                x_0 = states[:, 0]
            )

            loss = F.mse_loss(preds, states)

            optimizer.zero_grad()
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_loss += loss.item()

            if use_wandb and accelerator.is_main_process:
                wandb.log({'train/loss': loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        accelerator.print(f'epoch {epoch+1}/{epochs}, average loss: {avg_loss:.4f}', flush=True)

        if use_wandb and accelerator.is_main_process:
            wandb.log({'train/epoch_loss': avg_loss, 'epoch': epoch + 1})

    # evaluation

    accelerator.print('\nevaluating kalmanformer...\n')
    model.eval()

    with torch.no_grad():
        kf_test_preds = model(
            test_obs,
            F = None if learn_dynamics else test_F,
            H = None if learn_dynamics else H,
            x_0 = test_states[:, 0]
        )
        kf_mse = F.mse_loss(kf_test_preds, test_states)

    accelerator.print(f'kalmanformer test mse: {kf_mse.item():.4f}')
    accelerator.print(f'ekf test mse:          {ekf_mse.item():.4f}')
    accelerator.print('\n')

    if use_wandb and accelerator.is_main_process:
        wandb.log({'test/kf_mse': kf_mse.item()})

if __name__ == '__main__':
    fire.Fire(train)
