"""GPU check: in-place RoPE matches the float32 reference."""

import pytest
import torch

from tinyserve.kernels.reference import apply_rope as apply_rope_ref
from tinyserve.kernels.rope import apply_rope_
from tinyserve.model.rope import compute_inv_freq

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


@pytest.mark.parametrize(
    ("tokens", "q_heads", "kv_heads", "head_dim", "dtype"),
    [
        (1, 4, 2, 16, torch.float32),
        (3, 8, 2, 64, torch.float32),
        (2, 1, 1, 128, torch.float32),
        (1, 32, 8, 64, torch.bfloat16),
        (8, 32, 8, 64, torch.bfloat16),
        (4, 32, 8, 128, torch.bfloat16),
        (5, 4, 2, 16, torch.bfloat16),
    ],
)
def test_triton_rope_matches_reference(
    tokens: int, q_heads: int, kv_heads: int, head_dim: int, dtype: torch.dtype
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    inv_freq = compute_inv_freq(head_dim, 500000.0).to(device)
    positions = torch.arange(tokens, device=device)
    if tokens >= 1:
        positions[0] = 0
    if tokens >= 2:
        positions[1] = 8193
    if tokens >= 3:
        positions[-1] = 20000
    query = torch.randn(tokens, q_heads, head_dim, device=device, dtype=dtype)
    key = torch.randn(tokens, kv_heads, head_dim, device=device, dtype=dtype)
    expected_q = apply_rope_ref(query, positions, inv_freq)
    expected_k = apply_rope_ref(key, positions, inv_freq)
    apply_rope_(query, key, positions, inv_freq)
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    else:
        atol, rtol = 1e-2, 1e-2
    assert torch.allclose(query, expected_q, atol=atol, rtol=rtol)
    assert torch.allclose(key, expected_k, atol=atol, rtol=rtol)
