"""Paged KV cache storage and the startup memory profile.

A contiguous cache reserves every position up to a fixed maximum, including
positions a short sequence never uses. Pages are fixed-size blocks, so the
cache can hold many short sequences instead of a few padded ones. This module
decides how many blocks fit after weights and one profiled forward, then
allocates K and V. The block manager hands those blocks out; nothing here
tracks which sequence owns which block.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from tinyserve.config import CacheConfig


def dtype_nbytes(dtype: torch.dtype) -> int:
    """Storage size of one element. bf16 and fp16 are 2 bytes; float32 is 4."""
    return torch.empty((), dtype=dtype).element_size()


def bytes_per_token(num_layers: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:
    """Bytes of K and V, every layer, for one token.

    The 8B reference in the spec is 32 layers, 8 KV heads, head dim 128, bf16:
    131072 bytes, so 1 GiB holds 8192 tokens.
    """
    return num_layers * num_kv_heads * head_dim * 2 * dtype_nbytes(dtype)


def kv_budget_bytes(
    *,
    total_gpu_bytes: int,
    gpu_memory_utilization: float,
    weight_bytes: int,
    activation_bytes: int,
    safety_margin_bytes: int,
) -> int:
    """Bytes left for KV after weights, the profiled activation peak, and the margin.

    The spec's budget is ``total * utilization - weights - peak activations``.
    ``CacheConfig.memory_safety_margin_gib`` is held back as well: one profile
    forward can underestimate the activation peak of a real batch.
    """
    if min(total_gpu_bytes, weight_bytes, activation_bytes, safety_margin_bytes) < 0:
        raise ValueError("memory sizes must be >= 0")
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    usable = int(total_gpu_bytes * gpu_memory_utilization)
    return usable - weight_bytes - activation_bytes - safety_margin_bytes


def num_blocks_for_budget(kv_budget: int, bytes_per_block: int) -> int:
    """Whole blocks that fit in ``kv_budget``. A shortfall of one block is zero blocks."""
    if bytes_per_block <= 0:
        raise ValueError("bytes_per_block must be positive")
    if kv_budget < bytes_per_block:
        return 0
    return kv_budget // bytes_per_block


def weight_bytes(module: nn.Module) -> int:
    """Parameter storage, counting a tied embedding and LM head once.

    ``parameters()`` yields the shared tensor twice. Adding both would shrink
    the KV budget by a whole vocabulary matrix that is not actually allocated.
    """
    seen: set[int] = set()
    total = 0
    for param in module.parameters():
        ptr = param.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += param.nbytes
    return total


def activation_bytes_from_peak(baseline: int, peak: int) -> int:
    """Bytes allocated during a forward, above the memory already held by weights."""
    if peak < baseline:
        raise ValueError("peak allocation is below the baseline")
    return peak - baseline


def measure_activation_bytes(fn: Callable[[], None], device: torch.device) -> int:
    """Peak CUDA bytes ``fn`` allocates above the current baseline.

    CPU has no peak allocator stat. Returning 0 there would hand the whole
    budget to the cache and then OOM on the first real batch, so CPU callers
    pass a fake or previously measured ``activation_bytes`` instead.
    """
    if device.type != "cuda":
        raise RuntimeError("activation profiling needs CUDA; pass activation_bytes on CPU")
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return activation_bytes_from_peak(baseline, peak)


@dataclass
class CacheProfile:
    """What startup prints: how many blocks fit, and how many tokens that is."""

    num_blocks: int
    block_size: int
    capacity_tokens: int
    bytes_per_block: int
    bytes_per_token: int
    weight_bytes: int
    activation_bytes: int
    safety_margin_bytes: int
    total_gpu_bytes: int
    gpu_memory_utilization: float
    kv_budget_bytes: int
    used_override: bool

    def __str__(self) -> str:
        return (
            f"KV cache: {self.num_blocks} blocks, {self.capacity_tokens} tokens "
            f"({self.bytes_per_block} bytes/block)"
        )


def cache_profile(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    block_size: int,
    total_gpu_bytes: int,
    gpu_memory_utilization: float,
    weight_bytes: int,
    activation_bytes: int,
    safety_margin_bytes: int,
    num_blocks_override: int | None = None,
) -> CacheProfile:
    """Turn memory figures into a block count. An override skips the budget."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    per_token = bytes_per_token(num_layers, num_kv_heads, head_dim, dtype)
    per_block = per_token * block_size
    budget = kv_budget_bytes(
        total_gpu_bytes=total_gpu_bytes,
        gpu_memory_utilization=gpu_memory_utilization,
        weight_bytes=weight_bytes,
        activation_bytes=activation_bytes,
        safety_margin_bytes=safety_margin_bytes,
    )
    if num_blocks_override is None:
        blocks = num_blocks_for_budget(budget, per_block)
        used_override = False
    else:
        if num_blocks_override <= 0:
            raise ValueError("num_blocks_override must be positive")
        blocks = num_blocks_override
        used_override = True
    return CacheProfile(
        num_blocks=blocks,
        block_size=block_size,
        capacity_tokens=blocks * block_size,
        bytes_per_block=per_block,
        bytes_per_token=per_token,
        weight_bytes=weight_bytes,
        activation_bytes=activation_bytes,
        safety_margin_bytes=safety_margin_bytes,
        total_gpu_bytes=total_gpu_bytes,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_budget_bytes=budget,
        used_override=used_override,
    )


class PagedKVCache:
    """K and V for every layer, laid out as blocks of ``block_size`` tokens.

    Layer ``i`` is the spec's per-layer tensor
    ``[num_blocks, block_size, num_kv_heads, head_dim]``.
    A token at logical position ``p`` will live in ``block_table[p // block_size]`` at
    offset ``p % block_size``. This object does not store the block table.
    """

    def __init__(
        self,
        *,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        # Zeros so an unwritten slot cannot leak a stale value into attention.
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)

    @property
    def capacity_tokens(self) -> int:
        """Token slots in the whole pool, not the length of one sequence."""
        return self.num_blocks * self.block_size

    def layer_kv(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-layer views. Each is ``[num_blocks, block_size, num_kv_heads, head_dim]``."""
        return self.k[layer], self.v[layer]


def _safety_margin_bytes(cache_config: CacheConfig) -> int:
    return int(cache_config.memory_safety_margin_gib * (1024**3))


def _profile_prefill(model: nn.Module, num_tokens: int) -> None:
    """One prefill with no KV cache, so the peak is activations rather than a cache buffer."""
    if num_tokens < 1:
        raise ValueError("max_num_batched_tokens must be at least 1")
    device = next(model.parameters()).device
    input_ids = torch.zeros(1, num_tokens, dtype=torch.long, device=device)  # [1, num_tokens]
    model(input_ids, cache=None)


def build_paged_cache(
    model: nn.Module,
    cache_config: CacheConfig,
    *,
    total_gpu_bytes: int,
    max_num_batched_tokens: int,
    activation_bytes: int | None = None,
) -> tuple[PagedKVCache, CacheProfile]:
    """Allocate the paged cache. Print ``str(profile)`` at process startup.

    Pass ``activation_bytes`` to skip the profiling forward (unit tests, and
    CPU, where there is no peak allocator stat). On CUDA, leaving it ``None``
    runs one prefill of ``max_num_batched_tokens``. ``num_gpu_blocks_override``
    skips that forward and uses the given block count.
    """
    config = model.config
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if activation_bytes is None:
        if cache_config.num_gpu_blocks_override is None:
            activation_bytes = measure_activation_bytes(
                lambda: _profile_prefill(model, max_num_batched_tokens),
                device,
            )
        else:
            activation_bytes = 0
    profile = cache_profile(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=dtype,
        block_size=cache_config.block_size,
        total_gpu_bytes=total_gpu_bytes,
        gpu_memory_utilization=cache_config.gpu_memory_utilization,
        weight_bytes=weight_bytes(model),
        activation_bytes=activation_bytes,
        safety_margin_bytes=_safety_margin_bytes(cache_config),
        num_blocks_override=cache_config.num_gpu_blocks_override,
    )
    if profile.num_blocks < 1:
        raise RuntimeError(
            f"KV budget is {profile.kv_budget_bytes} bytes, which is below one block "
            f"of {profile.bytes_per_block} bytes"
        )
    cache = PagedKVCache(
        num_blocks=profile.num_blocks,
        block_size=profile.block_size,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=dtype,
        device=device,
    )
    return cache, profile
