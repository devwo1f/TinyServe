"""In-place rotary embedding from token positions.

Each program owns one head of one token. It turns the position into a float32
angle and writes both halves of that head back into the same tensor. One
program per head keeps a decode step (one token, many heads) on many programs
instead of one program walking the heads in series. Cos and sin are cast to
the tensor dtype before the multiply, matching ``kernels.reference.apply_rope``.
"""

from __future__ import annotations

import torch

from tinyserve.kernels.reference import apply_rope as apply_rope_ref

_KERNEL = None


def algorithm_bytes(
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int:
    """Bytes if Q and K are each read once and written once.

    Positions are int64, one per token. ``inv_freq`` is one float32 vector of
    length ``head_dim / 2``, shared by every head.
    """
    if num_tokens < 0 or num_heads < 1 or num_kv_heads < 1 or head_dim < 2 or head_dim % 2:
        raise ValueError("tokens >= 0, heads >= 1, and head_dim must be a positive even integer")
    item = torch.empty((), dtype=dtype).element_size()
    query = num_tokens * num_heads * head_dim * item
    key = num_tokens * num_kv_heads * head_dim * item
    positions = num_tokens * 8
    inv_freq = (head_dim // 2) * 4
    return 2 * (query + key) + positions + inv_freq


def apply_rope_(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q and K in place. CUDA launches the kernel; CPU copies the reference in.

    Both tensors are ``[num_tokens, num_heads, head_dim]``. Head counts may
    differ. The returned objects are the same storage the caller passed.
    """
    _check_pair(query, key, positions, inv_freq)
    if query.device.type != "cuda":
        query.copy_(apply_rope_ref(query, positions, inv_freq))
        key.copy_(apply_rope_ref(key, positions, inv_freq))
        return query, key
    _apply_cuda(query, positions, inv_freq)
    _apply_cuda(key, positions, inv_freq)
    return query, key


def _check_pair(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> None:
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("query and key must be [num_tokens, num_heads, head_dim]")
    if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
        raise ValueError("query and key must share num_tokens and head_dim")
    if query.dtype != key.dtype:
        raise ValueError("query and key dtypes differ")
    if positions.shape != (query.shape[0],):
        raise ValueError("positions must be [num_tokens]")
    half = query.shape[2] // 2
    if query.shape[2] % 2 != 0 or inv_freq.shape != (half,):
        raise ValueError("head_dim must be even and inv_freq must be [head_dim / 2]")


def _apply_cuda(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> None:
    num_tokens = x.shape[0]
    if num_tokens == 0:
        return
    flat = x if x.is_contiguous() else x.contiguous()
    pos = positions.to(device=x.device, dtype=torch.int64).contiguous()
    inv = inv_freq.to(device=x.device, dtype=torch.float32).contiguous()
    kernel, out_dtype = _kernel_and_dtype(x.dtype)
    kernel[(num_tokens, x.shape[1])](
        flat,
        pos,
        inv,
        x.shape[1],
        x.shape[2],
        OUT=out_dtype,
        BLOCK=64,
    )
    if flat.data_ptr() != x.data_ptr():
        x.copy_(flat)


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
    """Compile the head kernel once. The dtype is a launch argument."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _rope_kernel(
        x_ptr,
        pos_ptr,
        inv_ptr,
        n_heads,
        n_dim,
        OUT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        half = n_dim // 2
        base = x_ptr + (token * n_heads + head) * n_dim
        pos = tl.load(pos_ptr + token).to(tl.float32)
        for start in range(0, half, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < half
            first = tl.load(base + offs, mask=mask, other=0.0).to(OUT)
            second = tl.load(base + half + offs, mask=mask, other=0.0).to(OUT)
            inv = tl.load(inv_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            angle = pos * inv
            cos = tl.cos(angle).to(OUT)
            sin = tl.sin(angle).to(OUT)
            rotated_first = first * cos - second * sin
            rotated_second = second * cos + first * sin
            tl.store(base + offs, rotated_first, mask=mask)
            tl.store(base + half + offs, rotated_second, mask=mask)

    _KERNEL = _rope_kernel
    return _KERNEL
