"""A request and the sampling settings attached to it.

The fields are the contract in spec Section 9. The scheduler (Phase 4) is what
moves a sequence between waiting, running, and finished; this module only holds
the data.
"""

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class SamplingParams:
    """How to draw the next token, and when to stop.

    `temperature` 0 means greedy (argmax). `top_k` -1 means the cutoff is off.
    `stop_token_ids` is empty by default so benchmarks can run a fixed length.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 256
    seed: int | None = None
    stop_token_ids: list[int] = field(default_factory=list)


class SequenceStatus(Enum):
    """Where a request is in the engine. Preempted sequences are recomputed later."""

    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


@dataclass
class Sequence:
    """One generation request.

    `num_computed_tokens` is how much of the KV cache is already filled.
    """

    seq_id: int
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    sampling_params: SamplingParams
    status: SequenceStatus
    num_computed_tokens: int
    block_table: list[int]
    arrival_time: float
    first_token_time: float | None
    num_draft_tokens_proposed: int = 0
    num_draft_tokens_accepted: int = 0
