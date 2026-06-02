import pytest
import torch
from kalmanformer import KalmanFormer

@pytest.mark.parametrize('learn_dynamics', [False, True])
@pytest.mark.parametrize('use_memory', [False, True])
@pytest.mark.parametrize('use_callables', [False, True])
def test_kalmanformer(learn_dynamics, use_memory, use_callables):
    model = KalmanFormer(
        state_dim = 3,
        obs_dim = 3,
        dim = 16,
        depth = 2,
        heads = 2,
        dim_head = 8,
        use_memory = use_memory,
        learn_dynamics = learn_dynamics
    )

    batch, seq_len = 2, 5
    obs = torch.randn(batch, seq_len, 3)

    if use_callables:
        F = lambda step: torch.eye(3)
        H = lambda step: torch.eye(3)
    else:
        F = torch.eye(3)
        H = torch.eye(3)

    preds = model(obs, F = F, H = H)

    assert preds.shape == (batch, seq_len, 3)
