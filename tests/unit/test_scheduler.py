"""Scheduler: decodes first, then prefill chunks, within budget and free blocks."""

from tinyserve.config import SchedulerConfig
from tinyserve.engine.scheduler import Scheduler
from tinyserve.engine.sequence import SamplingParams, Sequence, SequenceStatus
from tinyserve.kv.block_manager import BlockManager


def _seq(seq_id: int, prompt_len: int, max_tokens: int = 4) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(range(prompt_len)),
        output_token_ids=[],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens),
        status=SequenceStatus.WAITING,
        num_computed_tokens=0,
        block_table=[],
        arrival_time=float(seq_id),
        first_token_time=None,
    )


def _cfg(**overrides) -> SchedulerConfig:
    values = {"max_num_batched_tokens": 8, "max_num_seqs": 4, "enable_chunked_prefill": True}
    values.update(overrides)
    return SchedulerConfig(**values)


def _advance(seq: Sequence, batch_item_tokens: int, *, decode: bool) -> None:
    """Stand in for the forward: KV for the scheduled tokens is now written."""
    seq.num_computed_tokens += batch_item_tokens
    if decode:
        seq.output_token_ids.extend([0] * batch_item_tokens)


def test_decodes_run_before_a_prefill_and_the_budget_is_the_sum():
    manager = BlockManager(8, 16)
    sched = Scheduler(manager, _cfg(max_num_batched_tokens=3))
    decode = _seq(0, 4)
    manager.allocate(decode, 4)
    decode.num_computed_tokens = 4
    decode.status = SequenceStatus.RUNNING
    sched.running.append(decode)
    waiting = _seq(1, 10)
    sched.add(waiting)

    batch = sched.schedule()

    assert [item.seq.seq_id for item in batch.seqs] == [0, 1]
    assert [item.is_prefill for item in batch.seqs] == [False, True]
    assert [item.num_new_tokens for item in batch.seqs] == [1, 2]
    assert batch.num_batched_tokens == 3
    assert batch.preempted == []
    assert decode.num_computed_tokens == 4
    assert waiting.status == SequenceStatus.RUNNING


def test_a_prompt_longer_than_the_budget_is_split_across_steps():
    manager = BlockManager(4, 16)
    sched = Scheduler(manager, _cfg(max_num_batched_tokens=4))
    sched.add(_seq(0, 10, max_tokens=1))

    first = sched.schedule()
    assert first.seqs[0].num_new_tokens == 4
    assert first.seqs[0].is_prefill
    _advance(first.seqs[0].seq, 4, decode=False)

    second = sched.schedule()
    assert second.seqs[0].num_new_tokens == 4
    _advance(second.seqs[0].seq, 4, decode=False)

    third = sched.schedule()
    assert third.seqs[0].num_new_tokens == 2
    assert third.seqs[0].is_prefill
    _advance(third.seqs[0].seq, 2, decode=False)

    decode = sched.schedule()
    assert [(item.num_new_tokens, item.is_prefill) for item in decode.seqs] == [(1, False)]


def test_chunking_off_skips_a_prompt_that_does_not_fit_in_one_step():
    manager = BlockManager(4, 16)
    sched = Scheduler(manager, _cfg(max_num_batched_tokens=4, enable_chunked_prefill=False))
    long = _seq(0, 10)
    short = _seq(1, 3)
    sched.add(long)
    sched.add(short)

    batch = sched.schedule()

    assert batch.seqs == []
    assert list(sched.waiting) == [long, short]
    assert long.block_table == []


def test_two_short_prefills_share_one_step():
    manager = BlockManager(4, 16)
    sched = Scheduler(manager, _cfg(max_num_batched_tokens=8, enable_chunked_prefill=False))
    sched.add(_seq(0, 3))
    sched.add(_seq(1, 4))

    batch = sched.schedule()

    assert [item.num_new_tokens for item in batch.seqs] == [3, 4]
    assert batch.num_batched_tokens == 7
    assert len(sched.running) == 2
    assert len(sched.waiting) == 0


def test_the_sequence_cap_leaves_the_next_request_waiting():
    manager = BlockManager(8, 16)
    sched = Scheduler(manager, _cfg(max_num_seqs=1, max_num_batched_tokens=32))
    sched.add(_seq(0, 2))
    sched.add(_seq(1, 2))

    batch = sched.schedule()

    assert [item.seq.seq_id for item in batch.seqs] == [0]
    assert [seq.seq_id for seq in sched.waiting] == [1]


def test_a_running_chunk_continues_before_a_new_request_is_admitted():
    manager = BlockManager(8, 16)
    sched = Scheduler(manager, _cfg(max_num_batched_tokens=4, max_num_seqs=2))
    sched.add(_seq(0, 10))
    sched.add(_seq(1, 10))

    first = sched.schedule()
    assert [item.seq.seq_id for item in first.seqs] == [0]
    _advance(first.seqs[0].seq, 4, decode=False)

    second = sched.schedule()
    assert [item.seq.seq_id for item in second.seqs] == [0]
    assert [seq.seq_id for seq in sched.waiting] == [1]


def test_a_tight_decode_budget_rotates_so_the_third_sequence_runs_next():
    manager = BlockManager(8, 16)
    sched = Scheduler(manager, _cfg(max_num_batched_tokens=2))
    for seq_id in (0, 1, 2):
        seq = _seq(seq_id, 1, max_tokens=3)
        manager.allocate(seq, 1)
        seq.num_computed_tokens = 1
        seq.status = SequenceStatus.RUNNING
        sched.running.append(seq)

    first = sched.schedule()
    assert [item.seq.seq_id for item in first.seqs] == [0, 1]
    for item in first.seqs:
        _advance(item.seq, 1, decode=True)

    second = sched.schedule()
    assert [item.seq.seq_id for item in second.seqs] == [2, 0]


def test_a_chunk_stops_at_the_last_free_block_and_does_not_preempt():
    manager = BlockManager(1, 4)
    sched = Scheduler(manager, _cfg(max_num_batched_tokens=100))
    sched.add(_seq(0, 10))

    first = sched.schedule()
    assert first.seqs[0].num_new_tokens == 4
    assert first.preempted == []
    _advance(first.seqs[0].seq, 4, decode=False)

    second = sched.schedule()
    assert second.seqs == []
    assert second.preempted == []
    assert first.seqs[0].seq.status == SequenceStatus.RUNNING


def test_a_decode_that_needs_a_new_block_is_skipped_when_the_pool_is_full():
    manager = BlockManager(1, 16)
    sched = Scheduler(manager, _cfg())
    seq = _seq(0, 16, max_tokens=2)
    manager.allocate(seq, 16)
    seq.num_computed_tokens = 16
    seq.status = SequenceStatus.RUNNING
    sched.running.append(seq)

    batch = sched.schedule()

    assert batch.seqs == []
    assert batch.preempted == []
    assert len(seq.block_table) == 1


def test_prefix_tokens_already_computed_are_not_prefilled_again():
    manager = BlockManager(4, 4)
    sched = Scheduler(manager, _cfg(max_num_batched_tokens=8))
    seq = _seq(0, 6)
    manager.allocate(seq, 4)
    seq.num_computed_tokens = 4
    sched.add(seq)

    batch = sched.schedule()

    assert batch.seqs[0].num_new_tokens == 2
    assert batch.seqs[0].is_prefill
    assert seq.num_computed_tokens == 4
