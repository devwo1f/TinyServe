"""Paged attention matches contiguous attention, including a short query over a prefix."""

import pytest
import torch

from tinyserve.kernels.kv_store import slot_mapping
from tinyserve.kernels.reference import paged_attention, standard_attention


def _plant(
    cache: torch.Tensor, block_table: list[int], tokens: torch.Tensor, block_size: int
) -> None:
    """Write logical tokens [seq, Nk, D] through the same slot formula as the KV store."""
    slots = slot_mapping(block_table, torch.arange(tokens.shape[0]), block_size)
    flat = cache.view(-1, cache.shape[-2], cache.shape[-1])  # [num_blocks * block_size, Nk, D]
    flat[slots] = tokens


def _caches(
    num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (num_blocks, block_size, num_kv_heads, head_dim)
    # A non-zero fill makes an accidental read of an unused offset show up.
    return torch.full(shape, 9.0), torch.full(shape, -9.0)


def _logical(query, key, value, seq_len: int) -> torch.Tensor:
    """Contiguous-cache attention for one sequence. key/value are [seq, Nk, D]."""
    return standard_attention(
        query.unsqueeze(0),
        key[:seq_len].transpose(0, 1).unsqueeze(0),
        value[:seq_len].transpose(0, 1).unsqueeze(0),
    )[0]


def test_decode_matches_contiguous_attention_on_scrambled_blocks():
    block_size = 4
    num_heads, num_kv_heads, head_dim = 4, 2, 8
    torch.manual_seed(0)
    # 17 tokens is one past four blocks; 6 tokens fills one block and part of the next.
    lengths = [17, 6]
    tables = [[3, 1, 6, 0, 4], [2, 5]]
    q_len = 1
    query = torch.randn(2, num_heads, q_len, head_dim)
    keys = [torch.randn(length, num_kv_heads, head_dim) for length in lengths]
    values = [torch.randn(length, num_kv_heads, head_dim) for length in lengths]
    k_cache, v_cache = _caches(8, block_size, num_kv_heads, head_dim)
    for table, key, value in zip(tables, keys, values, strict=True):
        _plant(k_cache, table, key, block_size)
        _plant(v_cache, table, value, block_size)

    out = paged_attention(query, k_cache, v_cache, tables, lengths)
    for i, length in enumerate(lengths):
        expected = _logical(query[i], keys[i], values[i], length)
        assert torch.equal(out[i], expected)


def test_a_query_chunk_over_a_prefix_matches_and_hides_future_keys():
    block_size = 4
    num_heads, num_kv_heads, head_dim = 4, 2, 8
    torch.manual_seed(1)
    # Context 13, query 3: the queries are positions 10, 11, and 12.
    length = 13
    q_len = 3
    table = [7, 2, 4, 1]
    query = torch.randn(1, num_heads, q_len, head_dim)
    key = torch.randn(length, num_kv_heads, head_dim)
    value = torch.randn(length, num_kv_heads, head_dim)
    k_cache, v_cache = _caches(8, block_size, num_kv_heads, head_dim)
    _plant(k_cache, table, key, block_size)
    _plant(v_cache, table, value, block_size)

    out = paged_attention(query, k_cache, v_cache, [table], [length])
    assert torch.equal(out[0], _logical(query[0], key, value, length))

    # Position 12 is a future key for the first query (position 10) and a real
    # key for the last query. Only the last query's output may change.
    moved = key.clone()
    moved[-1] = moved[-1] + 4
    k_moved, _ = _caches(8, block_size, num_kv_heads, head_dim)
    _plant(k_moved, table, moved, block_size)
    changed = paged_attention(query, k_moved, v_cache, [table], [length])
    assert torch.equal(changed[0, :, 0], out[0, :, 0])
    assert not torch.equal(changed[0, :, -1], out[0, :, -1])


def test_a_full_block_prefill_uses_the_causal_path():
    block_size = 4
    num_heads, num_kv_heads, head_dim = 2, 2, 4
    torch.manual_seed(2)
    length = 8
    table = [5, 1]
    query = torch.randn(1, num_heads, length, head_dim)
    key = torch.randn(length, num_kv_heads, head_dim)
    value = torch.randn(length, num_kv_heads, head_dim)
    k_cache, v_cache = _caches(6, block_size, num_kv_heads, head_dim)
    _plant(k_cache, table, key, block_size)
    _plant(v_cache, table, value, block_size)
    out = paged_attention(query, k_cache, v_cache, [table], [length])
    assert torch.equal(out[0], _logical(query[0], key, value, length))


def test_a_short_block_table_is_rejected():
    query = torch.randn(1, 2, 1, 4)
    k_cache, v_cache = _caches(4, block_size=4, num_kv_heads=2, head_dim=4)
    with pytest.raises(ValueError, match="block table"):
        paged_attention(query, k_cache, v_cache, [[0]], [5])
