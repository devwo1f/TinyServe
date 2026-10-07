"""Write new K and V into the paged cache.

The pool is addressed by a flat slot, not by sequence and position. A token at
logical position ``p`` sits in ``block_table[p // block_size]`` at offset
``p % block_size``, and the slot is ``block_id * block_size + offset``.
Indexing with that slot is the reference write. A later Triton kernel has to
match it; this file stays the check.
"""

from __future__ import annotations

import torch

from tinyserve.kv.cache import PagedKVCache


def slot_mapping(block_table: list[int], positions: torch.Tensor, block_size: int) -> torch.Tensor:
    """Flat cache slots for one sequence's new tokens.

    ``positions`` is ``[num_tokens]``. The block table is in logical order, so
    position 17 with block size 16 reads ``block_table[1]`` and offset 1.
    """
    if positions.ndim != 1:
        raise ValueError("positions must be [num_tokens]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if positions.numel() == 0:
        return positions.new_empty(0)
    if int(positions.min()) < 0:
        raise ValueError("positions must be >= 0")
    block_index = positions // block_size  # [num_tokens]
    offset = positions - block_index * block_size  # [num_tokens]
    if int(block_index.max()) >= len(block_table):
        raise ValueError("position falls past the block table")
    table = torch.as_tensor(block_table, dtype=torch.long, device=positions.device)
    block_id = table[block_index]  # [num_tokens]
    return block_id * block_size + offset  # [num_tokens]


def write_kv(
    cache: PagedKVCache,
    key: torch.Tensor,
    value: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Scatter one flattened batch of K/V into the pool.

    ``key`` and ``value`` are ``[num_layers, num_tokens, num_kv_heads, head_dim]``.
    That is the batch after sequences are concatenated, not ``[B, Nk, S, D]``.
    ``slots`` is ``[num_tokens]`` from ``slot_mapping``. The same slots are used
    for every layer, because each layer has its own copy of the pool.
    """
    if key.shape != value.shape:
        raise ValueError("key and value shapes differ")
    if key.ndim != 4:
        raise ValueError("key must be [num_layers, num_tokens, num_kv_heads, head_dim]")
    num_layers, num_tokens, num_kv_heads, head_dim = key.shape
    if num_layers != cache.num_layers:
        raise ValueError("key layers do not match the cache")
    if cache.k.shape[-2] != num_kv_heads or cache.k.shape[-1] != head_dim:
        raise ValueError("key heads do not match the cache")
    if slots.shape != (num_tokens,):
        raise ValueError("slots must be [num_tokens]")
    if num_tokens == 0:
        return
    nslots = cache.num_blocks * cache.block_size
    if int(slots.min()) < 0 or int(slots.max()) >= nslots:
        raise IndexError("slot is outside the paged cache")
    for layer in range(num_layers):
        # [num_blocks * block_size, num_kv_heads, head_dim]
        flat_k = cache.k[layer].view(-1, num_kv_heads, head_dim)
        flat_v = cache.v[layer].view(-1, num_kv_heads, head_dim)
        flat_k[slots] = key[layer]
        flat_v[slots] = value[layer]
