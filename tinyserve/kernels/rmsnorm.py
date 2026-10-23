"""Fused residual add and RMSNorm.

One program handles one row. It adds ``x`` and ``residual`` in float32, writes
that sum back in the model dtype, and writes the normalized row. The variance
uses the float32 sum, same as ``kernels.reference.add_rmsnorm``.

The row is loaded twice: once to finish the variance, then again to scale.
Keeping a 4096-wide row in registers spills, so the second load is the
readable form. ``issued_bytes`` counts both loads. ``algorithm_bytes`` counts
each tensor once, which is the traffic a fused kernel has to move.
"""

from __future__ import annotations

import torch

from tinyserve.kernels.reference import add_rmsnorm as add_rmsnorm_ref

_KERNEL = None


def algorithm_bytes(rows: int, hidden: int, dtype: torch.dtype) -> int:
    """Bytes if each value is read or written once.

    That is ``x``, ``residual``, and ``weight`` in, and the sum plus the
    normalized row out. ``weight`` is one vector, shared by every row.
    """
    _check_shape(rows, hidden)
    item = torch.empty((), dtype=dtype).element_size()
    return (4 * rows * hidden + hidden) * item


def issued_bytes(rows: int, hidden: int, dtype: torch.dtype) -> int:
    """Global loads and stores the kernel issues, counting the second pass.

    Each row loads ``x`` and ``residual`` twice and ``weight`` once, then
    stores the sum and the normalized row.
    """
    _check_shape(rows, hidden)
    item = torch.empty((), dtype=dtype).element_size()
    return 7 * rows * hidden * item


def add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual add and RMSNorm. CUDA uses the Triton kernel; CPU uses the reference.

    The two paths share one contract, so a CPU test of the reference is also
    the spec for the kernel. Triton is imported only when a CUDA tensor shows
    up, because the CPU environment does not ship it.
    """
    if x.device.type != "cuda":
        return add_rmsnorm_ref(x, residual, weight, eps)
    return _add_rmsnorm_cuda(x, residual, weight, eps)


def _check_shape(rows: int, hidden: int) -> None:
    if rows < 0 or hidden < 1:
        raise ValueError("rows must be >= 0 and hidden must be >= 1")


def _add_rmsnorm_cuda(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or residual.shape != x.shape:
        raise ValueError("x and residual must both be [rows, hidden]")
    if weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("weight must be [hidden]")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if x.dtype != residual.dtype:
        raise ValueError("x and residual dtypes differ")
    rows, _hidden = x.shape
    if rows == 0:
        return x.clone(), residual.clone()  # [0, hidden]
    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()
    normed = torch.empty_like(x)  # [rows, hidden]
    residual_out = torch.empty_like(x)  # [rows, hidden]
    kernel, out_dtype = _kernel_and_dtype(x.dtype)
    kernel[(rows,)](
        x,
        residual,
        weight,
        normed,
        residual_out,
        x.shape[1],
        float(eps),
        OUT=out_dtype,
        BLOCK=256,
    )
    return normed, residual_out


def _kernel_and_dtype(dtype: torch.dtype):
    import triton.language as tl

    table = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }
    if dtype not in table:
        raise ValueError(f"unsupported dtype {dtype}")
    return _kernel(), table[dtype]


def _kernel():
    """Compile the row kernel once. The dtype is a launch argument."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _add_rmsnorm_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        out_ptr,
        residual_out_ptr,
        n_cols,
        eps,
        OUT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        x_row = x_ptr + row * n_cols
        residual_row = residual_ptr + row * n_cols
        out_row = out_ptr + row * n_cols
        residual_out_row = residual_out_ptr + row * n_cols
        sum_sq = tl.zeros((), dtype=tl.float32)
        for start in range(0, n_cols, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < n_cols
            x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            residual = tl.load(residual_row + cols, mask=mask, other=0.0).to(tl.float32)
            summed = x + residual
            sum_sq += tl.sum(summed * summed, axis=0)
            tl.store(residual_out_row + cols, summed.to(OUT), mask=mask)
        variance = sum_sq / n_cols
        rstd = tl.rsqrt(variance + eps)
        for start in range(0, n_cols, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < n_cols
            x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            residual = tl.load(residual_row + cols, mask=mask, other=0.0).to(tl.float32)
            weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            normed = (x + residual) * rstd * weight
            tl.store(out_row + cols, normed.to(OUT), mask=mask)

    _KERNEL = _add_rmsnorm_kernel
    return _KERNEL
