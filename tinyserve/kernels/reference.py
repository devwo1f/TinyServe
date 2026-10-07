"""Slow, correct attention over a paged KV cache.

A later Triton kernel has to match this file. The check is the contiguous
cache: gather each sequence's pages back into logical order, then run the
same attention the dense cache uses. In float32 the outputs match.

The query tokens are already in the cache. They occupy the last ``q_len``
positions of ``seq_lens``. A decode step has ``q_len`` 1. A chunked prefill
or a speculative verification has ``q_len`` greater than 1 and a prefix
before those positions.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    """True where a query may attend. Queries sit at the end of the key sequence.

    Shape ``[q_len, k_len]``. Query local index ``i`` is absolute position
    ``k_len - q_len + i``. The contiguous model uses this same mask: a short
    query has to see the end of the cache, not the start.
    """
    q_pos = torch.arange(k_len - q_len, k_len, device=device)[:, None]  # [q_len, 1]
    k_pos = torch.arange(k_len, device=device)[None, :]  # [1, k_len]
    return k_pos <= q_pos


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat each KV head across the query heads that share it.

    ``x`` is ``[batch, num_kv_heads, seq, head_dim]``. Grouped-query attention
    has several query heads read one KV head, so the KV tensor is repeated
    before the score. ``n_rep`` 1 is the multi-head case and returns ``x``.
    """
    if n_rep == 1:
        return x
    batch, n_kv, seq, dim = x.shape
    expanded = x[:, :, None, :, :].expand(batch, n_kv, n_rep, seq, dim)
    return expanded.reshape(batch, n_kv * n_rep, seq, dim)  # [batch, num_heads, seq, head_dim]


def standard_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Attention for queries that sit at the end of a contiguous context.

    ``query`` is ``[batch, num_heads, q_len, head_dim]``. ``key`` and ``value``
    are ``[batch, num_kv_heads, k_len, head_dim]`` with ``k_len >= q_len``.
    The three cases match ``Attention.forward``: a full causal sequence, a
    single new token that can see every key, and a short query over a longer
    cache. Sharing those cases is what makes the paged path match the
    contiguous cache in float32.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be rank-4")
    if key.shape != value.shape:
        raise ValueError("key and value shapes differ")
    if query.shape[0] != key.shape[0] or query.shape[3] != key.shape[3]:
        raise ValueError("query batch or head dim does not match key")
    num_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    if num_kv_heads == 0 or num_heads % num_kv_heads != 0:
        raise ValueError("query heads must be a multiple of KV heads")
    q_len = query.shape[2]
    k_len = key.shape[2]
    if q_len > k_len:
        raise ValueError("query is longer than the context")
    n_rep = num_heads // num_kv_heads
    key = repeat_kv(key, n_rep)  # [batch, num_heads, k_len, head_dim]
    value = repeat_kv(value, n_rep)
    if q_len == k_len:
        out = nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
    elif q_len == 1:
        out = nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)
    else:
        out = nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=_causal_mask(q_len, k_len, query.device)
        )
    return out  # [batch, num_heads, q_len, head_dim]


def gather_paged_kv(cache: torch.Tensor, block_table: Sequence[int], seq_len: int) -> torch.Tensor:
    """Logical K or V for one sequence, dropping the unused tail of the last block.

    ``cache`` is ``[num_blocks, block_size, num_kv_heads, head_dim]``. The
    block table is in logical order and is not padded: block 0 is a real
    block, so a pad of 0 would be read as data. Returns
    ``[seq_len, num_kv_heads, head_dim]``.
    """
    if cache.ndim != 4:
        raise ValueError("cache must be [num_blocks, block_size, num_kv_heads, head_dim]")
    if seq_len < 0:
        raise ValueError("seq_len must be >= 0")
    block_size = cache.shape[1]
    num_kv_heads = cache.shape[2]
    head_dim = cache.shape[3]
    if seq_len == 0:
        return cache.new_empty((0, num_kv_heads, head_dim))
    if block_size < 1:
        raise ValueError("block_size must be positive")
    num_needed = (seq_len + block_size - 1) // block_size
    if len(block_table) < num_needed:
        raise ValueError("block table is shorter than the context")
    ids = [int(block_table[i]) for i in range(num_needed)]
    if min(ids) < 0 or max(ids) >= cache.shape[0]:
        raise IndexError("block id is outside the paged cache")
    index = torch.tensor(ids, dtype=torch.long, device=cache.device)
    pages = cache.index_select(0, index)  # [num_needed, block_size, num_kv_heads, head_dim]
    flat = pages.reshape(num_needed * block_size, num_kv_heads, head_dim)
    return flat[:seq_len].contiguous()  # [seq_len, num_kv_heads, head_dim]


def _seq_lens(seq_lens: Sequence[int] | torch.Tensor, num_seqs: int) -> list[int]:
    if isinstance(seq_lens, torch.Tensor):
        if seq_lens.shape != (num_seqs,):
            raise ValueError("seq_lens must be [num_seqs]")
        values = [int(x) for x in seq_lens.tolist()]
    else:
        if len(seq_lens) != num_seqs:
            raise ValueError("seq_lens must be [num_seqs]")
        values = [int(x) for x in seq_lens]
    if any(length < 0 for length in values):
        raise ValueError("seq_lens must be >= 0")
    return values


def paged_attention(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: Sequence[Sequence[int]],
    seq_lens: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Gather each sequence, then run ``standard_attention``.

    ``query`` is ``[num_seqs, num_heads, q_len, head_dim]``. Every sequence in
    the call has that same query length; a mixed decode and prefill batch is
    two calls. ``seq_lens[i]`` is the context length after the new tokens are
    stored, and those new tokens are the last ``q_len`` positions.
    """
    if query.ndim != 4:
        raise ValueError("query must be [num_seqs, num_heads, q_len, head_dim]")
    if k_cache.shape != v_cache.shape:
        raise ValueError("K and V caches differ")
    if k_cache.ndim != 4:
        raise ValueError("cache must be [num_blocks, block_size, num_kv_heads, head_dim]")
    if query.shape[3] != k_cache.shape[3]:
        raise ValueError("query head dim does not match the cache")
    if query.dtype != k_cache.dtype or query.device != k_cache.device:
        raise ValueError("query and cache must share dtype and device")
    num_seqs = query.shape[0]
    q_len = query.shape[2]
    if len(block_tables) != num_seqs:
        raise ValueError("one block table per sequence")
    lengths = _seq_lens(seq_lens, num_seqs)
    outputs: list[torch.Tensor] = []
    for i, length in enumerate(lengths):
        if length < q_len:
            raise ValueError("context is shorter than the query")
        key = gather_paged_kv(k_cache, block_tables[i], length)  # [length, Nk, D]
        value = gather_paged_kv(v_cache, block_tables[i], length)
        # SDPA wants heads before the sequence, one sequence at a time.
        key = key.transpose(0, 1).unsqueeze(0)  # [1, Nk, length, D]
        value = value.transpose(0, 1).unsqueeze(0)
        outputs.append(standard_attention(query[i : i + 1], key, value))
    if not outputs:
        return query.new_empty(query.shape)
    return torch.cat(outputs, dim=0)  # [num_seqs, num_heads, q_len, head_dim]


def block_table_ids(row: torch.Tensor) -> list[int]:
    """Drop a trailing ``-1`` pad. Block 0 is a real block, so the pad is not 0.

    A ``-1`` in the middle is a hole, not padding, and is rejected.
    """
    ids = [int(block_id) for block_id in row.tolist()]
    if -1 not in ids:
        return ids
    first_pad = ids.index(-1)
    if any(block_id != -1 for block_id in ids[first_pad:]):
        raise ValueError("block table pad must be trailing -1s")
    return ids[:first_pad]


def paged_attention_flat(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Paged attention for a flattened batch whose sequences have different query lengths.

    ``query`` is ``[num_tokens, num_heads, head_dim]``. ``query_start_loc`` is
    the prefix sum of those lengths, shape ``[num_seqs + 1]``. ``block_tables``
    is ``[num_seqs, max_blocks]`` padded with ``-1``. Each sequence's query is
    the last ``q_len`` tokens of its context, already written into the cache.
    """
    if query.ndim != 3:
        raise ValueError("query must be [num_tokens, num_heads, head_dim]")
    num_tokens = query.shape[0]
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 1:
        raise ValueError("query_start_loc must be [num_seqs + 1]")
    num_seqs = query_start_loc.shape[0] - 1
    if block_tables.shape[0] != num_seqs or seq_lens.shape != (num_seqs,):
        raise ValueError("block tables and seq_lens must have one row per sequence")
    if int(query_start_loc[0]) != 0 or int(query_start_loc[-1]) != num_tokens:
        raise ValueError("query_start_loc must cover every query token")
    pieces: list[torch.Tensor] = []
    for index in range(num_seqs):
        start = int(query_start_loc[index])
        end = int(query_start_loc[index + 1])
        if end <= start:
            raise ValueError("each sequence needs at least one query token")
        # [1, num_heads, q_len, head_dim]
        one_query = query[start:end].transpose(0, 1).unsqueeze(0).contiguous()
        attended = paged_attention(
            one_query,
            k_cache,
            v_cache,
            [block_table_ids(block_tables[index])],
            [int(seq_lens[index])],
        )
        pieces.append(attended.squeeze(0).transpose(0, 1))  # [q_len, num_heads, head_dim]
    return torch.cat(pieces, dim=0)  # [num_tokens, num_heads, head_dim]
