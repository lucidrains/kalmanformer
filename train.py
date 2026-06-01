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
from typing import Literal
from scipy.integrate import odeint

import torch
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.optim import Adam

import fire
import wandb
from accelerate import Accelerator

from kalmanformer import KalmanFormer, ExtendedKalmanFilter

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

def generate_dataset(
    num_trajectories = 100,
    T = 5.,
    dt = 0.05,
    process_noise_std = 0.05,
    obs_noise_std = 0.05,
    pseudo_linear = True
):
    t = np.arange(0, T, dt)
    dataset_states = []
    dataset_obs = []
    dataset_F = []

    for _ in range(num_trajectories):
        x0 = np.random.uniform(-10, 10, size=(3,))
        states = odeint(lorenz_deriv, x0, t)

        observations = states + np.random.normal(0, obs_noise_std, size=states.shape)
        F_seq = np.array([get_jacobian(state, dt, pseudo_linear=pseudo_linear) for state in states])

        dataset_states.append(states)
        dataset_obs.append(observations)
        dataset_F.append(F_seq)

    return (
        tensor(np.stack(dataset_states), dtype = torch.float32),
        tensor(np.stack(dataset_obs), dtype = torch.float32),
        tensor(np.stack(dataset_F), dtype = torch.float32)
    )

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
    cpu: bool = False,
    base_estimator_type: Literal['none', 'kf', 'ekf'] = 'none',
    pseudo_linear: bool = True
):
    assert base_estimator_type in ('none', 'kf', 'ekf'), "base_estimator_type must be 'none', 'kf', or 'ekf'"

    accelerator = Accelerator(cpu = cpu)
    device = accelerator.device

    if use_wandb and accelerator.is_main_process:
        wandb.init(project = 'kalmanformer', config = locals())

    accelerator.print('generating dataset...', flush=True)

    kwargs = dict(num_trajectories = num_train_trajectories, T = sequence_length, dt = dt, pseudo_linear = pseudo_linear)
    train_states, train_obs, train_F = [t.to(device) for t in generate_dataset(**kwargs)]

    kwargs.update(num_trajectories = num_test_trajectories)
    test_states, test_obs, test_F = [t.to(device) for t in generate_dataset(**kwargs)]

    H = torch.eye(3, device = device)
    Q = torch.eye(3, device = device) * 0.8**2
    R = torch.eye(3, device = device) * 1.0**2

    origin_state = np.zeros(3)
    fixed_F = tensor(get_jacobian(origin_state, dt, pseudo_linear = pseudo_linear), dtype = torch.float32, device = device)

    use_fixed_F = base_estimator_type == 'kf'

    # baseline EKF

    accelerator.print('\nrunning ekf baseline...\n')
    ekf_model = ExtendedKalmanFilter(state_dim = 3, obs_dim = 3, Q = Q, R = R).to(device)
    ekf_test_preds = ekf_model(test_obs, test_F, H, x_0 = test_states[:, 0])
    ekf_mse = F.mse_loss(ekf_test_preds, test_states)
    accelerator.print(f'ekf test mse: {ekf_mse.item():.4f}')

    if use_wandb and accelerator.is_main_process:
        wandb.log({'test/ekf_mse': ekf_mse.item()})

    # baseline KF

    accelerator.print('\nrunning kf baseline...\n')
    kf_model = ExtendedKalmanFilter(state_dim = 3, obs_dim = 3, Q = Q, R = R).to(device)
    kf_test_preds = kf_model(test_obs, fixed_F, H, x_0 = test_states[:, 0])
    kf_mse = F.mse_loss(kf_test_preds, test_states)
    accelerator.print(f'kf test mse: {kf_mse.item():.4f}')

    if not use_kalmanformer:
        return

    # kalmanformer

    if base_estimator_type == 'none':
        base_model = None
    else:
        base_model = ExtendedKalmanFilter(state_dim = 3, obs_dim = 3, Q = Q, R = R)

    accelerator.print(f'\ntraining kalmanformer (base={base_estimator_type}, memory={use_memory})...\n')

    model = KalmanFormer(
        state_dim = 3,
        obs_dim = 3,
        dim = 64,
        depth = 2,
        heads = 2,
        learn_dynamics = learn_dynamics,
        use_memory = use_memory,
        base_estimator = base_model
    )

    optimizer = Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

    dataset = TensorDataset(train_states, train_obs, train_F)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for states, obs, F_seq in dataloader:

            F_arg = None if learn_dynamics else (fixed_F if use_fixed_F else F_seq)
            H_arg = None if learn_dynamics else H

            preds = model(
                obs,
                F = F_arg,
                H = H_arg,
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
        F_arg = None if learn_dynamics else (fixed_F if use_fixed_F else test_F)
        H_arg = None if learn_dynamics else H

        test_preds = model(
            test_obs,
            F = F_arg,
            H = H_arg,
            x_0 = test_states[:, 0]
        )
        test_mse = F.mse_loss(test_preds, test_states)

    accelerator.print(f'kalmanformer (base={base_estimator_type}) test mse: {test_mse.item():.4f}')
    accelerator.print(f'ekf test mse: {ekf_mse.item():.4f}')
    accelerator.print(f'kf test mse:  {kf_mse.item():.4f}')
    accelerator.print('')

    if use_wandb and accelerator.is_main_process:
        wandb.log({'test/kalmanformer_mse': test_mse.item()})

if __name__ == '__main__':
    fire.Fire(train)
