"""Generation on one paged pool, several requests per step.

``step`` asks the scheduler who runs, forwards that mixed batch once, samples
the next token for every sequence whose context is now full, and frees
sequences that stopped. A short decode is in the same forward as a prefill
chunk, instead of waiting for some other request to finish.

The pool is sized for every prompt in the call. Full prompt blocks can stay
cached after a request finishes, so a later call that starts with the same
tokens skips that prefill. A call that needs more blocks builds a new pool
and drops the cache.
"""

import time
from dataclasses import dataclass, field

import torch

from tinyserve.config import SchedulerConfig
from tinyserve.engine.model_runner import prepare_model_input, run_paged
from tinyserve.engine.sampler import sample_token
from tinyserve.engine.scheduler import ScheduledBatch, ScheduledSeq, Scheduler
from tinyserve.engine.sequence import SamplingParams, Sequence, SequenceStatus
from tinyserve.kv.block_manager import BlockManager, blocks_for_tokens
from tinyserve.kv.cache import PagedKVCache
from tinyserve.kv.prefix_cache import PrefixCache
from tinyserve.kv.waste import SlotSample, sample_slots
from tinyserve.model.llama import LlamaForCausalLM
from tinyserve.model.tokenizer import Tokenizer


@dataclass
class GenerationResult:
    """The finished output of one prompt.

    `text` is a full detokenize of `output_token_ids`. Streaming holds back a
    trailing partial character; this offline call has the whole sequence, so
    it does not.
    """

    prompt_token_ids: list[int]
    output_token_ids: list[int]
    text: str
    num_computed_tokens: int
    # Prompt tokens whose KV was already in the prefix cache. The rest of the
    # prompt was prefilled on this request.
    num_cached_prompt_tokens: int = 0
    # Seconds from the start of this request. None when no token was emitted.
    # The last cache write, after the final token already exists, is not included.
    ttft_s: float | None = None
    e2e_s: float = 0.0
    itl_s: list[float] | None = None
    # One entry per model step when recording is on. Empty otherwise.
    # Captured before free() clears the block table.
    slot_samples: tuple[SlotSample, ...] = ()


@dataclass
class _Live:
    """Bookkeeping ``schedule`` does not own.

    ``pending`` is the logit row from the forward that filled the context.
    The next decode samples it. A preemption drops the row: that KV is gone.
    """

    seq: Sequence
    generator: torch.Generator | None
    arrival: float
    cached: int
    token_times: list[float] = field(default_factory=list)
    samples: list[SlotSample] = field(default_factory=list)
    pending: torch.Tensor | None = None


class _EvictingBlockManager:
    """Reclaim parked prefix blocks before the scheduler preempts anyone.

    ``can_allocate`` is the scheduler's signal that the free list is short.
    A cached block with no reader is not in use. Taking it back is cheaper
    than dropping a running request and prefilling it again.
    """

    def __init__(self, manager: BlockManager, prefix: PrefixCache | None):
        self.manager = manager
        self._prefix = prefix

    def can_allocate(self, seq: Sequence, num_new_tokens: int) -> bool:
        if self.manager.can_allocate(seq, num_new_tokens):
            return True
        if self._prefix is None:
            return False
        while not self.manager.can_allocate(seq, num_new_tokens):
            evicted = self._prefix.evict_lru()
            if evicted is None:
                return False
            self.manager.reclaim(evicted)
        return True

    def allocate(self, seq: Sequence, num_new_tokens: int) -> None:
        self.manager.allocate(seq, num_new_tokens)

    def free(self, seq: Sequence) -> None:
        self.manager.free(seq)


class Engine:
    """Run ``LlamaForCausalLM`` on a paged KV cache, many requests per step.

    ``generate`` is the spec's offline API. ``generate_tokens`` is the same
    loop for callers who already have ids (a chat template, or a test).
    ``step`` is one iteration: schedule, run, sample, update, free finished.
    ``block_size`` is the page size. The default matches ``CacheConfig``.
    Prefix caching is on by default for the same reason. ``record_kv_waste``
    keeps a slot sample after every forward. ``contiguous_max_len`` is the
    dense reservation those samples charge; it defaults to the model's context.
    ``num_blocks`` fixes the pool size so a test can force preemption. Left
    unset, the pool holds every request in the call at once.
    """

    def __init__(
        self,
        model: LlamaForCausalLM,
        tokenizer: Tokenizer,
        block_size: int = 16,
        enable_prefix_caching: bool = True,
        record_kv_waste: bool = False,
        contiguous_max_len: int | None = None,
        scheduler_config: SchedulerConfig | None = None,
        num_blocks: int | None = None,
    ):
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if contiguous_max_len is not None and contiguous_max_len < 1:
            raise ValueError("contiguous_max_len must be positive")
        if num_blocks is not None and num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.record_kv_waste = record_kv_waste
        self.contiguous_max_len = contiguous_max_len
        self.scheduler_config = scheduler_config or SchedulerConfig()
        self.num_blocks = num_blocks
        self._next_seq_id = 0
        self._prefix: PrefixCache | None = None
        self._manager: BlockManager | None = None
        self._cache: PagedKVCache | None = None
        self._scheduler: Scheduler | None = None
        self._live: dict[int, _Live] = {}

    def generate(
        self, prompts: list[str], sampling_params: SamplingParams
    ) -> list[GenerationResult]:
        """Complete the prompts on one shared pool.

        `encode` does not add a beginning-of-sequence token. A chat prompt
        should go through `Tokenizer.apply_chat_template` and then
        `generate_tokens`, or the positions will not match the template.
        """
        if not isinstance(sampling_params, SamplingParams):
            raise TypeError("sampling_params must be one SamplingParams for the whole call")
        encoded = [self.tokenizer.encode(prompt) for prompt in prompts]
        return self.generate_tokens(encoded, sampling_params)

    @torch.inference_mode()
    def generate_tokens(
        self, prompts: list[list[int]], sampling_params: SamplingParams
    ) -> list[GenerationResult]:
        """Greedy or sampled completion for every prompt, batched each step."""
        if not isinstance(sampling_params, SamplingParams):
            raise TypeError("sampling_params must be one SamplingParams for the whole call")
        if sampling_params.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if not prompts:
            return []
        self._live = {}
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        self._ensure_pool(self._blocks_for(prompts, sampling_params), device, dtype)
        if self._scheduler is None:
            raise RuntimeError("the KV pool is missing")
        if self._scheduler.running or self._scheduler.waiting:
            raise RuntimeError("a previous call left sequences in the scheduler")
        sequences = [self._enqueue(prompt_ids, sampling_params, device) for prompt_ids in prompts]
        # Preemption can recompute a request. The cap is far above one forward
        # per token, and it exists so a stuck pool fails instead of spinning.
        cap = sum(len(prompt) + sampling_params.max_tokens for prompt in prompts) * len(prompts) * 8
        steps = 0
        while any(seq.status != SequenceStatus.FINISHED for seq in sequences):
            batch = self.step()
            steps += 1
            if steps > max(cap, 1):
                raise RuntimeError("engine step did not finish the batch")
            if not batch.seqs and not batch.preempted:
                raise RuntimeError("no token was scheduled; pool or token budget is too small")
        return [self._result(seq) for seq in sequences]

    @torch.inference_mode()
    def step(self) -> ScheduledBatch:
        """One iteration: schedule, run, sample, update, free finished.

        Decode entries are sampled before the forward. The scheduler already
        reserved a slot for that new token, and the model can only run a token
        whose id is on the sequence. A prefill chunk runs tokens that already
        exist. Logits are kept only when the chunk lands on the last context
        token; a mid-prompt logit repeats a token the prompt already has.
        """
        if self._scheduler is None or self._manager is None or self._cache is None:
            raise RuntimeError("the KV pool is missing")
        self._restore_prefixes()
        batch = self._scheduler.schedule()
        for victim in batch.preempted:
            live = self._live.get(victim.seq_id)
            if live is not None:
                live.pending = None
        runnable: list[ScheduledSeq] = []
        for item in batch.seqs:
            if item.is_prefill:
                runnable.append(item)
                continue
            if self._accept_new_token(item.seq):
                runnable.append(item)
        if runnable:
            self._forward(runnable)
            self._note_slots(runnable)
        self._retire()
        return batch

    def _blocks_for(self, prompts: list[list[int]], params: SamplingParams) -> int:
        """How many blocks this call needs if every request is live together."""
        if self.num_blocks is not None:
            return self.num_blocks
        return sum(
            blocks_for_tokens(len(prompt_ids) + params.max_tokens, self.block_size)
            for prompt_ids in prompts
        )

    def _ensure_pool(self, num_blocks: int, device: torch.device, dtype: torch.dtype) -> None:
        """Allocate the shared KV pool. A bigger later request drops the old pages."""
        if self._manager is not None and self._manager.num_blocks >= num_blocks:
            return
        prefix = PrefixCache(self.block_size) if self.enable_prefix_caching else None
        self._prefix = prefix
        self._manager = BlockManager(
            num_blocks,
            self.block_size,
            is_cached=None if prefix is None else prefix.is_cached,
            park_cached=None if prefix is None else prefix.park,
        )
        self._scheduler = Scheduler(
            _EvictingBlockManager(self._manager, prefix), self.scheduler_config
        )
        config = self.model.config
        self._cache = PagedKVCache(
            num_blocks=num_blocks,
            block_size=self.block_size,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=dtype,
            device=device,
        )

    def _enqueue(
        self, prompt_ids: list[int], params: SamplingParams, device: torch.device
    ) -> Sequence:
        """Queue one prompt. Cached full blocks are shared before the first schedule."""
        if not prompt_ids:
            raise ValueError("prompt is empty")
        limit = self.model.config.max_position_embeddings
        if len(prompt_ids) + params.max_tokens > limit:
            raise ValueError(
                f"prompt ({len(prompt_ids)}) plus max_tokens ({params.max_tokens}) "
                f"exceeds max_position_embeddings ({limit})"
            )
        if self._scheduler is None:
            raise RuntimeError("the KV pool is missing")
        sequence = Sequence(
            seq_id=self._next_seq_id,
            prompt_token_ids=list(prompt_ids),
            output_token_ids=[],
            sampling_params=params,
            status=SequenceStatus.WAITING,
            num_computed_tokens=0,
            block_table=[],
            arrival_time=time.perf_counter(),
            first_token_time=None,
        )
        self._next_seq_id += 1
        cached = self._take_cached_prefix(sequence, prompt_ids)
        self._live[sequence.seq_id] = _Live(
            seq=sequence,
            generator=_request_generator(params, device),
            arrival=sequence.arrival_time,
            cached=cached,
        )
        self._scheduler.add(sequence)
        return sequence

    def _restore_prefixes(self) -> None:
        """A preempted request lost its table. Share a cached prompt again if one exists."""
        if self._scheduler is None:
            return
        for seq in self._scheduler.waiting:
            if seq.num_computed_tokens != 0 or seq.block_table:
                continue
            cached = self._take_cached_prefix(seq, seq.prompt_token_ids)
            live = self._live.get(seq.seq_id)
            if live is not None:
                live.cached = cached

    def _take_cached_prefix(self, sequence: Sequence, prompt_ids: list[int]) -> int:
        """Share cached full blocks and return how many prompt tokens that covers."""
        if self._prefix is None or self._manager is None:
            return 0
        reused = self._prefix.match(prompt_ids)
        for block_id in reused:
            self._manager.share(sequence, block_id)
            self._prefix.touch(block_id)
        sequence.num_computed_tokens = len(reused) * self.block_size
        return sequence.num_computed_tokens

    def _accept_new_token(self, seq: Sequence) -> bool:
        """Sample the token the last forward already scored. False means stop.

        The clock is read after ``.item()``, which waits until those logits
        are ready. The forward that stores this token happens afterwards, so
        the timestamp does not include that write.
        """
        live = self._live[seq.seq_id]
        if live.pending is None:
            raise RuntimeError(f"sequence {seq.seq_id} has no logits to sample")
        logits = live.pending  # [vocab]
        live.pending = None
        token = int(sample_token(logits, seq.sampling_params, live.generator).item())
        if token in set(seq.sampling_params.stop_token_ids):
            # schedule() reserved a slot for this token. It will not be written.
            if self._manager is None:
                raise RuntimeError("the KV pool is missing")
            self._manager.truncate(seq, seq.num_computed_tokens)
            self._mark_finished(seq)
            return False
        now = time.perf_counter()
        if seq.first_token_time is None:
            seq.first_token_time = now
        live.token_times.append(now)
        seq.output_token_ids.append(token)
        return True

    def _forward(self, items: list[ScheduledSeq]) -> None:
        """Write KV for the scheduled tokens and keep logits that can be sampled."""
        if self._cache is None or self._manager is None:
            raise RuntimeError("the KV pool is missing")
        seqs = [item.seq for item in items]
        counts = [item.num_new_tokens for item in items]
        model_input = prepare_model_input(seqs, counts, self.block_size, self._cache.k.device)
        logits = run_paged(self.model, self._cache, model_input)  # [num_seqs, vocab]
        _sync(self._cache.k.device)
        for index, item in enumerate(items):
            seq = item.seq
            seq.num_computed_tokens += item.num_new_tokens
            context = _context_len(seq)
            if seq.num_computed_tokens < context:
                continue
            if seq.num_computed_tokens > context:
                raise RuntimeError("computed tokens ran past the sequence")
            if len(seq.output_token_ids) >= seq.sampling_params.max_tokens:
                self._mark_finished(seq)
                continue
            self._live[seq.seq_id].pending = logits[index].clone()  # [vocab]

    def _note_slots(self, ran: list[ScheduledSeq]) -> None:
        """One waste sample for the sequences that still hold blocks, before free."""
        if not self.record_kv_waste or self._scheduler is None:
            return
        max_len = self.contiguous_max_len
        if max_len is None:
            max_len = self.model.config.max_position_embeddings
        live = [seq for seq in self._scheduler.running if seq.block_table]
        if not live:
            return
        sample = sample_slots(
            [list(seq.block_table) for seq in live],
            [seq.num_computed_tokens for seq in live],
            block_size=self.block_size,
            max_len=max_len,
        )
        for item in ran:
            self._live[item.seq.seq_id].samples.append(sample)

    def _mark_finished(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.FINISHED
        live = self._live.get(seq.seq_id)
        if live is not None:
            live.pending = None

    def _retire(self) -> None:
        """Cache full prompt blocks, free KV, and leave the running queue."""
        if self._scheduler is None or self._manager is None:
            return
        for seq in list(self._scheduler.running):
            if seq.status != SequenceStatus.FINISHED:
                continue
            if self._prefix is not None:
                self._prefix.cache_prompt(seq.block_table, seq.prompt_token_ids)
            self._manager.free(seq)
            self._scheduler.finish(seq)

    def _result(self, seq: Sequence) -> GenerationResult:
        live = self._live[seq.seq_id]
        times = live.token_times
        finished = time.perf_counter()
        ttft = None if not times else times[0] - live.arrival
        e2e = (times[-1] if times else finished) - live.arrival
        itl = [times[i] - times[i - 1] for i in range(1, len(times))]
        return GenerationResult(
            prompt_token_ids=seq.prompt_token_ids,
            output_token_ids=list(seq.output_token_ids),
            text=self.tokenizer.decode(seq.output_token_ids),
            num_computed_tokens=seq.num_computed_tokens,
            num_cached_prompt_tokens=live.cached,
            ttft_s=ttft,
            e2e_s=e2e,
            itl_s=itl,
            slot_samples=tuple(live.samples),
        )


def _context_len(seq: Sequence) -> int:
    """Prompt plus tokens already generated. Both need KV before the next sample."""
    return len(seq.prompt_token_ids) + len(seq.output_token_ids)


def _sync(device: torch.device) -> None:
    """Wait until queued GPU work for this request has finished."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _request_generator(params: SamplingParams, device: torch.device) -> torch.Generator | None:
    """A generator owned by this request. `None` leaves the global RNG alone.

    Greedy sampling ignores it. A seed still builds one so a later temperature
    change on the same params object does not silently share process state.
    """
    if params.seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(params.seed)
    return generator
