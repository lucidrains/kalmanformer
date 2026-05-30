import pytest
import torch
from kalmanformer import KalmanFormer

@pytest.mark.parametrize('use_memory', [False, True])
def test_kalmanformer(use_memory):
    model = KalmanFormer(
        state_dim = 3,
        obs_dim = 3,
        dim = 16,
        depth = 2,
        heads = 2,
        dim_head = 8,
        use_memory = use_memory
    )

    batch = 2
    seq_len = 5

    obs = torch.randn(batch, seq_len, 3)
    F = torch.eye(3)
    H = torch.eye(3)

    preds = model(obs, F = F, H = H)

    assert preds.shape == (batch, seq_len, 3)
