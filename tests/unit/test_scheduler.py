"""Scheduler: decodes first, then prefill chunks, within budget and free blocks."""

from collections import deque

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


def test_finish_drops_the_sequence_and_leaves_its_blocks_allocated():
    manager = BlockManager(4, 4)
    sched = Scheduler(manager, _cfg())
    seq = _running(manager, _seq(0, 4), 4)
    sched.running.append(seq)

    sched.finish(seq)

    assert seq.status == SequenceStatus.FINISHED
    assert list(sched.running) == []
    assert seq.block_table != []
    assert sched.schedule().seqs == []


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


def test_the_newest_running_sequence_is_preempted_when_blocks_run_out():
    manager = BlockManager(2, 4)
    sched = Scheduler(manager, _cfg(max_num_batched_tokens=1, max_num_seqs=2))
    older = _running(manager, _seq(0, 4, max_tokens=2), 4)
    newer = _running(manager, _seq(1, 3, max_tokens=2), 4)
    newer.output_token_ids = [9]
    older.arrival_time = 1.0
    newer.arrival_time = 2.0
    sched.running.append(older)
    sched.running.append(newer)

    batch = sched.schedule()

    assert [item.seq.seq_id for item in batch.seqs] == [0]
    assert [seq.seq_id for seq in batch.preempted] == [1]
    assert newer.status == SequenceStatus.PREEMPTED
    assert newer.num_computed_tokens == 0
    assert newer.output_token_ids == [9]
    assert newer.block_table == []
    assert list(sched.waiting) == [newer]
    assert len(older.block_table) == 2


def test_preemption_still_finishes_every_request_with_the_same_tokens():
    # Each request peaks at 2 blocks. The small pool holds one of them.
    small = _finish_all(num_blocks=2)
    roomy = _finish_all(num_blocks=8)
    assert small == roomy
    assert [len(tokens) for tokens in small] == [4, 4]


def _running(manager: BlockManager, seq: Sequence, num_tokens: int) -> Sequence:
    manager.allocate(seq, num_tokens)
    seq.num_computed_tokens = num_tokens
    seq.status = SequenceStatus.RUNNING
    return seq


def _finish_all(num_blocks: int) -> list[list[int]]:
    """Drive schedule until both requests hit max_tokens. Finished KV is freed here.

    The engine step loop (P4.3) is what will do that free. This stand-in is
    enough to show a squeezed pool still emits the same tokens.
    """
    manager = BlockManager(num_blocks, 4)
    sched = Scheduler(manager, _cfg(max_num_batched_tokens=32, max_num_seqs=4))
    seqs = [_seq(0, 4, max_tokens=4), _seq(1, 4, max_tokens=4)]
    for seq in seqs:
        sched.add(seq)
    for _ in range(200):
        if all(seq.status == SequenceStatus.FINISHED for seq in seqs):
            break
        batch = sched.schedule()
        if not batch.seqs and not batch.preempted:
            break
        for item in batch.seqs:
            _play(item.seq, item.num_new_tokens)
            if len(item.seq.output_token_ids) >= item.seq.sampling_params.max_tokens:
                item.seq.status = SequenceStatus.FINISHED
                manager.free(item.seq)
                sched.running = deque(seq for seq in sched.running if seq.seq_id != item.seq.seq_id)
    assert [seq.status for seq in seqs] == [SequenceStatus.FINISHED, SequenceStatus.FINISHED]
    return [list(seq.output_token_ids) for seq in seqs]


def _play(seq: Sequence, num_new_tokens: int) -> None:
    """Write KV. Sample only positions that are not already in the context."""
    start = seq.num_computed_tokens
    for offset in range(num_new_tokens):
        pos = start + offset
        context = seq.prompt_token_ids + seq.output_token_ids
        if pos < len(context):
            continue
        seq.output_token_ids.append(sum(context) % 50)
    seq.num_computed_tokens = start + num_new_tokens
