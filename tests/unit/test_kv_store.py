"""Paged KV writes land on the block and offset the slot formula names."""

import pytest
import torch

from tinyserve.kernels.kv_store import slot_mapping, write_kv
from tinyserve.kv.cache import PagedKVCache


def _cache() -> PagedKVCache:
    return PagedKVCache(
        num_blocks=8,
        block_size=4,
        num_layers=2,
        num_kv_heads=2,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def test_slot_crosses_a_block_boundary_using_the_physical_id():
    # Logical blocks are physical ids 3 then 6, not 0 then 1.
    slots = slot_mapping([3, 6], torch.tensor([0, 3, 4]), block_size=4)
    # p 0 -> block 3 offset 0; p 3 -> block 3 offset 3; p 4 -> block 6 offset 0
    assert slots.tolist() == [3 * 4 + 0, 3 * 4 + 3, 6 * 4 + 0]


def test_values_land_in_that_block_and_offset_and_nowhere_else():
    cache = _cache()
    positions_a = torch.tensor([0, 3, 4])
    positions_b = torch.tensor([0])
    slots = torch.cat(
        [
            slot_mapping([3, 6], positions_a, 4),
            slot_mapping([1], positions_b, 4),
        ]
    )
    num_tokens = 4
    key = torch.arange(2 * num_tokens * 2 * 2, dtype=torch.float32).view(2, num_tokens, 2, 2)
    value = key + 100
    write_kv(cache, key, value, slots)

    assert torch.equal(cache.k[0, 3, 0], key[0, 0])
    assert torch.equal(cache.k[0, 3, 3], key[0, 1])
    assert torch.equal(cache.k[0, 6, 0], key[0, 2])
    assert torch.equal(cache.k[0, 1, 0], key[0, 3])
    assert torch.equal(cache.v[1, 6, 0], value[1, 2])
    # The other offsets of block 3, and an unused block, stay empty.
    assert torch.count_nonzero(cache.k[0, 3, 1]) == 0
    assert torch.count_nonzero(cache.k[0, 3, 2]) == 0
    assert torch.count_nonzero(cache.k[:, 0]) == 0
    assert torch.count_nonzero(cache.v[:, 2]) == 0


def test_position_past_the_table_and_a_negative_position_are_rejected():
    with pytest.raises(ValueError, match="past the block table"):
        slot_mapping([3], torch.tensor([4]), block_size=4)
    with pytest.raises(ValueError, match="positions"):
        slot_mapping([3], torch.tensor([-1]), block_size=4)


def test_empty_write_does_not_touch_the_cache():
    cache = _cache()
    key = torch.ones(2, 0, 2, 2)
    write_kv(cache, key, key, torch.zeros(0, dtype=torch.long))
    assert torch.count_nonzero(cache.k) == 0


def test_a_slot_outside_the_pool_is_rejected():
    cache = _cache()
    key = torch.ones(2, 1, 2, 2)
    with pytest.raises(IndexError, match="outside"):
        write_kv(cache, key, key, torch.tensor([8 * 4]))


def test_a_layer_count_mismatch_is_rejected():
    cache = _cache()
    key = torch.ones(1, 1, 2, 2)
    with pytest.raises(ValueError, match="layers"):
        write_kv(cache, key, key, torch.tensor([0]))
