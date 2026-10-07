"""Who runs on this step, and for how many tokens.

Decodes go first so a long prefill cannot stall a request that is already
generating. The leftover token budget is spent on prompts, in chunks when
chunking is on, so one prompt does not have to fit in a single step.

``schedule`` reserves KV blocks. It does not advance ``num_computed_tokens``.
That count means the forward has written KV, so the caller adds
``num_new_tokens`` after the forward. A second ``schedule`` before that add
would reserve the same tokens again. An empty free list does not preempt
anyone; that is P4.2. The sequence stays where it is and this step skips it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from tinyserve.config import SchedulerConfig
from tinyserve.engine.sequence import Sequence, SequenceStatus
from tinyserve.kv.block_manager import BlockManager


@dataclass
class ScheduledSeq:
    """One sequence in this step, and how many new tokens it may run."""

    seq: Sequence
    num_new_tokens: int  # 1 for decode, chunk length for prefill
    is_prefill: bool


@dataclass
class ScheduledBatch:
    """The step the runner will flatten into one forward."""

    seqs: list[ScheduledSeq]
    num_batched_tokens: int
    preempted: list[Sequence]


class Scheduler:
    """Waiting queue, running queue, and one ``schedule`` call per step.

    ``max_num_seqs`` is how many sequences may be in flight, not how many
    new ones a step may start. A sequence already running still gets its
    decode, or the next prefill chunk, when the token budget and the free
    list allow it.
    """

    def __init__(self, block_manager: BlockManager, config: SchedulerConfig):
        self._manager = block_manager
        self._config = config
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add(self, seq: Sequence) -> None:
        """Queue a request. KV is reserved later, inside ``schedule``."""
        if not seq.prompt_token_ids:
            raise ValueError("prompt is empty")
        if seq.status == SequenceStatus.FINISHED:
            raise ValueError("cannot queue a finished sequence")
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def schedule(self) -> ScheduledBatch:
        """Pick decodes, then prefill chunks, and reserve their blocks.

        Running prefills continue before a new request is admitted, so a
        chunk already in flight is not stuck behind a prompt that just
        arrived. A waiting request that does not fit stops the queue. The
        one behind it is not skipped.
        """
        chosen: list[ScheduledSeq] = []
        budget = self._config.max_num_batched_tokens
        decode_ids: list[int] = []

        for seq in list(self.running):
            if budget < 1:
                break
            if not _is_decode(seq):
                continue
            if not self._manager.can_allocate(seq, 1):
                continue
            self._manager.allocate(seq, 1)
            chosen.append(ScheduledSeq(seq, 1, False))
            budget -= 1
            decode_ids.append(seq.seq_id)

        for seq in list(self.running):
            if budget < 1:
                break
            if not _has_prefill_left(seq):
                continue
            chunk = self._chunk(seq, budget)
            if chunk < 1:
                continue
            self._reserve_prefill(seq, chunk, chosen)
            budget -= chunk

        while self.waiting and budget >= 1 and len(self.running) < self._config.max_num_seqs:
            seq = self.waiting[0]
            chunk = self._chunk(seq, budget)
            if chunk < 1:
                break
            self._reserve_prefill(seq, chunk, chosen)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            budget -= chunk

        self._rotate_decodes(decode_ids)
        return ScheduledBatch(
            seqs=chosen,
            num_batched_tokens=sum(item.num_new_tokens for item in chosen),
            preempted=[],
        )

    def _chunk(self, seq: Sequence, budget: int) -> int:
        """How many prompt tokens this step can run. Zero means not this step."""
        remaining = len(seq.prompt_token_ids) - seq.num_computed_tokens
        if remaining <= 0 or budget <= 0:
            return 0
        if not self._config.enable_chunked_prefill:
            if remaining > budget or not self._manager.can_allocate(seq, remaining):
                return 0
            return remaining
        return _tokens_that_fit(self._manager, seq, min(remaining, budget))

    def _reserve_prefill(self, seq: Sequence, chunk: int, chosen: list[ScheduledSeq]) -> None:
        self._manager.allocate(seq, chunk)
        chosen.append(ScheduledSeq(seq, chunk, True))

    def _rotate_decodes(self, decode_ids: list[int]) -> None:
        """Move decodes that ran to the back, so a tight budget still reaches the rest.

        Prefills stay in front of that rotation. The next step still considers
        every decode before any prefill; the rotation only changes which
        decode wins when they cannot all fit.
        """
        if not decode_ids:
            return
        ran = set(decode_ids)
        front: list[Sequence] = []
        back: list[Sequence] = []
        for seq in self.running:
            if seq.seq_id in ran:
                back.append(seq)
            else:
                front.append(seq)
        self.running = deque(front + back)


def _is_decode(seq: Sequence) -> bool:
    """True when the prompt KV is done and another output token is allowed."""
    if seq.status == SequenceStatus.FINISHED:
        return False
    if seq.num_computed_tokens < len(seq.prompt_token_ids):
        return False
    return len(seq.output_token_ids) < seq.sampling_params.max_tokens


def _has_prefill_left(seq: Sequence) -> bool:
    return seq.num_computed_tokens < len(seq.prompt_token_ids)


def _tokens_that_fit(manager: BlockManager, seq: Sequence, limit: int) -> int:
    """Largest prefix of ``limit`` tokens the free list can hold. Zero is allowed."""
    if limit <= 0:
        return 0
    if manager.can_allocate(seq, limit):
        return limit
    lo = 0
    hi = limit - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if manager.can_allocate(seq, mid):
            lo = mid
        else:
            hi = mid - 1
    return lo
