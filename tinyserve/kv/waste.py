"""KV memory waste for paged blocks versus a contiguous max-length reservation.

Spec Section 10 defines the fraction as (allocated slots - used slots) /
allocated slots, sampled once per step. Used slots are tokens whose K and V
are already written. Paged allocation rounds each sequence up to a whole
block, so the leftover in the last block is the waste. A contiguous
max-length cache instead reserves ``max_len`` slots for every live sequence
from the first step, including tokens that do not exist yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from tinyserve.engine.sequence import SamplingParams, Sequence, SequenceStatus
from tinyserve.kv.block_manager import BlockManager, blocks_for_tokens


def kv_memory_waste(allocated_slots: int, used_slots: int) -> float:
    """The Section 10 fraction. Zero when the step holds no slots.

    A used count past the allocation is a bug in the caller, not a negative
    waste. An empty allocation with nothing used is a step with no sequences.
    """
    if allocated_slots < 0 or used_slots < 0:
        raise ValueError("slot counts must be >= 0")
    if used_slots > allocated_slots:
        raise ValueError("used slots cannot exceed allocated slots")
    if allocated_slots == 0:
        return 0.0
    return (allocated_slots - used_slots) / allocated_slots


@dataclass(frozen=True)
class SlotSample:
    """One step: tokens written, slots the pages hold, and the dense reservation."""

    used_slots: int
    paged_allocated_slots: int
    contiguous_allocated_slots: int

    @property
    def paged_waste(self) -> float:
        return kv_memory_waste(self.paged_allocated_slots, self.used_slots)

    @property
    def contiguous_waste(self) -> float:
        return kv_memory_waste(self.contiguous_allocated_slots, self.used_slots)


def sample_slots(
    block_tables: list[list[int]],
    used_tokens: list[int],
    *,
    block_size: int,
    max_len: int,
) -> SlotSample:
    """Count slots for the sequences that are live on this step.

    Each sequence contributes ``len(block_table) * block_size`` paged slots
    and ``max_len`` contiguous slots. ``max_len`` is the serving limit, not
    the length of this sequence: that is the reservation a dense cache makes
    up front. Shared blocks are not in this measurement; each table is counted
    on its own.
    """
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if max_len < 1:
        raise ValueError("max_len must be positive")
    if len(block_tables) != len(used_tokens):
        raise ValueError("each sequence needs a block table and a used count")
    paged = 0
    used = 0
    for table, count in zip(block_tables, used_tokens, strict=True):
        if count < 0:
            raise ValueError("used slots must be >= 0")
        if count > max_len:
            raise ValueError("used slots exceed the contiguous max length")
        slots = len(table) * block_size
        if slots < count:
            raise ValueError("block table does not cover the used tokens")
        paged += slots
        used += count
    return SlotSample(
        used_slots=used,
        paged_allocated_slots=paged,
        contiguous_allocated_slots=len(used_tokens) * max_len,
    )


def request_step_samples(
    prompt_len: int,
    output_len: int,
    *,
    block_size: int,
    max_len: int,
) -> list[SlotSample]:
    """Samples for one request the way the engine steps, with caching off.

    Prefill is a single step that fills the prompt. Each output token is its
    own step. The block ids come from ``BlockManager``, so this stays aligned
    with ``allocate`` rather than a second copy of the rounding rule.
    """
    if prompt_len < 1 or output_len < 1:
        raise ValueError("prompt_len and output_len must be at least 1")
    total = prompt_len + output_len
    if total > max_len:
        raise ValueError("prompt plus output exceeds max_len")
    manager = BlockManager(blocks_for_tokens(total, block_size), block_size)
    sequence = _sequence(prompt_len, output_len)
    samples: list[SlotSample] = []
    manager.allocate(sequence, prompt_len)
    sequence.num_computed_tokens = prompt_len
    samples.append(_one(sequence, block_size, max_len))
    for _ in range(output_len):
        manager.allocate(sequence, 1)
        sequence.num_computed_tokens += 1
        samples.append(_one(sequence, block_size, max_len))
    return samples


def concurrent_slot_sample(
    lengths: list[int],
    *,
    block_size: int,
    max_len: int,
) -> SlotSample:
    """Every sequence live at once, each at its full length.

    This is the batch a contiguous allocator sizes to ``max_len`` per
    sequence. Paging only holds the blocks those lengths need. The engine
    still runs one request at a time; this snapshot is the same allocator
    asked to keep them all.
    """
    if any(length < 1 for length in lengths):
        raise ValueError("lengths must be at least 1")
    if any(length > max_len for length in lengths):
        raise ValueError("a length exceeds max_len")
    needed = sum(blocks_for_tokens(length, block_size) for length in lengths)
    if needed == 0:
        return sample_slots([], [], block_size=block_size, max_len=max_len)
    manager = BlockManager(needed, block_size)
    tables: list[list[int]] = []
    used: list[int] = []
    for index, length in enumerate(lengths):
        sequence = _sequence(length, 1)
        sequence.seq_id = index
        manager.allocate(sequence, length)
        sequence.num_computed_tokens = length
        tables.append(list(sequence.block_table))
        used.append(length)
    return sample_slots(tables, used, block_size=block_size, max_len=max_len)


def _one(sequence: Sequence, block_size: int, max_len: int) -> SlotSample:
    return sample_slots(
        [sequence.block_table],
        [sequence.num_computed_tokens],
        block_size=block_size,
        max_len=max_len,
    )


def _sequence(prompt_len: int, output_len: int) -> Sequence:
    return Sequence(
        seq_id=0,
        prompt_token_ids=[0] * prompt_len,
        output_token_ids=[],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=output_len),
        status=SequenceStatus.RUNNING,
        num_computed_tokens=0,
        block_table=[],
        arrival_time=0.0,
        first_token_time=None,
    )
