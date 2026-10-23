"""GPU check: the Triton fused norm matches the float32 reference."""

import pytest
import torch

from tinyserve.kernels.reference import add_rmsnorm as add_rmsnorm_ref
from tinyserve.kernels.rmsnorm import add_rmsnorm

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


@pytest.mark.parametrize(
    ("rows", "hidden", "dtype"),
    [
        (1, 1, torch.float32),
        (2, 17, torch.float32),
        (4, 64, torch.float32),
        (1, 2048, torch.bfloat16),
        (8, 2048, torch.bfloat16),
        (3, 4096, torch.bfloat16),
        (2, 17, torch.bfloat16),
    ],
)
def test_triton_matches_reference(rows: int, hidden: int, dtype: torch.dtype):
    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(rows, hidden, device=device, dtype=dtype)
    residual = torch.randn(rows, hidden, device=device, dtype=dtype)
    weight = torch.randn(hidden, device=device, dtype=dtype)
    eps = 1e-5
    normed, residual_out = add_rmsnorm(x, residual, weight, eps)
    expected_norm, expected_residual = add_rmsnorm_ref(x, residual, weight, eps)
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    else:
        # Spec Section 12: bf16 kernel vs reference.
        atol, rtol = 1e-2, 1e-2
    assert torch.allclose(residual_out, expected_residual, atol=atol, rtol=rtol)
    assert torch.allclose(normed, expected_norm, atol=atol, rtol=rtol)
