"""Block free list, reference counts, and truncate. No GPU."""

import pytest

from tinyserve.engine.sequence import SamplingParams, Sequence, SequenceStatus
from tinyserve.kv.block_manager import BlockManager, blocks_for_tokens


def _seq(seq_id: int = 0, computed: int = 0) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=[],
        output_token_ids=[],
        sampling_params=SamplingParams(),
        status=SequenceStatus.WAITING,
        num_computed_tokens=computed,
        block_table=[],
        arrival_time=0.0,
        first_token_time=None,
    )


def test_exact_block_boundaries():
    assert blocks_for_tokens(0, 16) == 0
    assert blocks_for_tokens(15, 16) == 1
    assert blocks_for_tokens(16, 16) == 1
    assert blocks_for_tokens(17, 16) == 2

    manager = BlockManager(8, 16)
    one_short = _seq()
    manager.allocate(one_short, 15)
    assert one_short.block_table == [0]

    exact = _seq(1)
    manager.allocate(exact, 16)
    assert exact.block_table == [1]

    one_past = _seq(2)
    manager.allocate(one_past, 17)
    assert one_past.block_table == [2, 3]

    exact.num_computed_tokens = 16
    manager.allocate(exact, 1)
    assert exact.block_table == [1, 4]

    partial = _seq(3)
    manager.allocate(partial, 15)
    partial.num_computed_tokens = 15
    manager.allocate(partial, 1)
    assert partial.block_table == [5]


def test_full_block_then_another_full_block_appends_one():
    manager = BlockManager(4, 16)
    seq = _seq()
    manager.allocate(seq, 16)
    seq.num_computed_tokens = 16
    assert manager.can_allocate(seq, 16)
    manager.allocate(seq, 16)
    assert seq.block_table == [0, 1]
    assert seq.num_computed_tokens == 16


def test_exhaustion_leaves_the_free_list_unchanged():
    manager = BlockManager(2, 16)
    seq = _seq()
    assert not manager.can_allocate(seq, 48)
    with pytest.raises(RuntimeError, match="need 3 KV blocks"):
        manager.allocate(seq, 48)
    assert seq.block_table == []
    assert manager.num_free_blocks() == 2


def test_free_returns_blocks_and_a_second_free_is_a_no_op():
    manager = BlockManager(4, 16)
    seq = _seq()
    manager.allocate(seq, 32)
    assert [manager.ref_count_of(block_id) for block_id in (0, 1)] == [1, 1]
    manager.free(seq)
    assert seq.block_table == []
    assert manager.num_free_blocks() == 4
    assert manager.ref_count_of(0) == 0
    manager.free(seq)
    assert manager.num_free_blocks() == 4


def test_shared_block_stays_allocated_until_both_sequences_free_it():
    manager = BlockManager(4, 16)
    first = _seq(0)
    second = _seq(1)
    manager.allocate(first, 16)
    manager.share(second, 0)
    assert manager.ref_count_of(0) == 2
    manager.free(first)
    assert manager.ref_count_of(0) == 1
    assert manager.num_free_blocks() == 3
    assert second.block_table == [0]
    manager.free(second)
    assert manager.ref_count_of(0) == 0
    assert manager.num_free_blocks() == 4


def test_truncate_drops_only_blocks_past_the_new_length():
    manager = BlockManager(4, 16)
    seq = _seq()
    manager.allocate(seq, 32)
    seq.num_computed_tokens = 32

    manager.truncate(seq, 17)
    assert seq.block_table == [0, 1]
    assert seq.num_computed_tokens == 17

    manager.truncate(seq, 16)
    assert seq.block_table == [0]
    assert manager.ref_count_of(1) == 0
    assert manager.num_free_blocks() == 3

    manager.truncate(seq, 15)
    assert seq.block_table == [0]

    manager.truncate(seq, 0)
    assert seq.block_table == []
    assert seq.num_computed_tokens == 0
    assert manager.num_free_blocks() == 4


def test_truncate_of_a_shared_tail_does_not_free_the_block():
    manager = BlockManager(4, 16)
    owner = _seq(0)
    sharer = _seq(1)
    manager.allocate(owner, 32)
    owner.num_computed_tokens = 32
    manager.share(sharer, 1)
    manager.truncate(owner, 16)
    assert owner.block_table == [0]
    assert manager.ref_count_of(1) == 1
    assert sharer.block_table == [1]
    assert manager.num_free_blocks() == 2


def test_truncate_cannot_grow_and_rejects_a_negative_length():
    manager = BlockManager(2, 16)
    seq = _seq()
    manager.allocate(seq, 16)
    with pytest.raises(ValueError, match="cannot grow"):
        manager.truncate(seq, 32)
    with pytest.raises(ValueError, match="new_length"):
        manager.truncate(seq, -1)


def test_cached_block_is_parked_instead_of_freed():
    parked: list[int] = []
    cached = {1}

    manager = BlockManager(
        4,
        16,
        is_cached=cached.__contains__,
        park_cached=parked.append,
    )
    seq = _seq()
    manager.allocate(seq, 32)
    manager.free(seq)
    assert parked == [1]
    assert manager.ref_count_of(1) == 0
    assert manager.num_free_blocks() == 3
    manager.reclaim(1)
    assert manager.num_free_blocks() == 4
    with pytest.raises(RuntimeError, match="already free"):
        manager.reclaim(1)


def test_cannot_share_a_free_block():
    manager = BlockManager(2, 16)
    with pytest.raises(ValueError, match="free block"):
        manager.share(_seq(), 0)
