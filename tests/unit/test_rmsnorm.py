"""Reference residual-add RMSNorm, and the byte counts the benchmark reports."""

import torch

from tinyserve.kernels.reference import add_rmsnorm
from tinyserve.kernels.rmsnorm import algorithm_bytes, issued_bytes
from tinyserve.model.llama import RMSNorm


def test_reference_matches_rmsnorm_module_in_float32():
    torch.manual_seed(0)
    rows, hidden = 3, 17
    x = torch.randn(rows, hidden)
    residual = torch.randn(rows, hidden)
    weight = torch.randn(hidden)
    eps = 1e-5
    normed, residual_out = add_rmsnorm(x, residual, weight, eps)
    module = RMSNorm(hidden, eps)
    with torch.no_grad():
        module.weight.copy_(weight)
    expected = module(x + residual)
    assert residual_out.shape == (rows, hidden)
    assert torch.equal(residual_out, x + residual)
    assert torch.allclose(normed, expected, atol=1e-6, rtol=1e-6)


def test_reference_one_column_and_a_zero_row_batch():
    weight = torch.tensor([2.0])
    x = torch.tensor([[3.0], [-4.0]])
    residual = torch.tensor([[1.0], [4.0]])
    normed, residual_out = add_rmsnorm(x, residual, weight, 1e-5)
    # sum is 4 and 0. mean square is 16 and 0. scale is 2 / sqrt(16+eps) and 0.
    assert torch.allclose(residual_out, torch.tensor([[4.0], [0.0]]))
    assert normed[1].item() == 0.0
    assert normed[0].item() > 0
    empty, empty_residual = add_rmsnorm(x[:0], residual[:0], weight, 1e-5)
    assert empty.shape == (0, 1)
    assert empty_residual.shape == (0, 1)


def test_byte_counts_match_the_kernel_loops():
    # algorithm: x, residual, residual_out, normed, plus one weight vector.
    assert algorithm_bytes(1, 4, torch.float32) == (4 * 1 * 4 + 4) * 4
    # issued: x and residual twice, weight once, two stores, all per row.
    assert issued_bytes(2, 4, torch.bfloat16) == 7 * 2 * 4 * 2
