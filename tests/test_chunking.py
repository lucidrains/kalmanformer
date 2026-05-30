import torch
from kalmanformer import KalmanFormer, ExtendedKalmanFilter

def test_chunking_invariance():
    b = 2
    seq_len = 10
    state_dim = 3
    obs_dim = 3

    model = KalmanFormer(
        state_dim = state_dim,
        obs_dim = obs_dim,
        dim = 64,
        depth = 1,
        heads = 2,
        use_memory = True
    )

    observations = torch.randn(b, seq_len, obs_dim)
    F = torch.randn(b, seq_len, state_dim, state_dim)
    H = torch.randn(b, seq_len, obs_dim, state_dim)
    x_0 = torch.randn(b, state_dim)

    # run full sequence
    model.eval()
    with torch.no_grad():
        out_full, state_full = model(
            observations, F, H, x_0 = x_0, return_state = True
        )

    # run in chunks
    chunk_size = 5
    observations_1 = observations[:, :chunk_size]
    F_1 = F[:, :chunk_size]
    H_1 = H[:, :chunk_size]

    observations_2 = observations[:, chunk_size:]
    F_2 = F[:, chunk_size-1 : -1]
    H_2 = H[:, chunk_size:]

    with torch.no_grad():
        out_chunk_1, state_1 = model(
            observations_1, F_1, H_1, x_0 = x_0, return_state = True
        )

        out_chunk_2, state_2 = model(
            observations_2, F_2, H_2, initial_state = state_1, return_state = True
        )

    out_chunks = torch.cat((out_chunk_1, out_chunk_2), dim = 1)

    assert torch.allclose(out_full, out_chunks, atol = 1e-5)
