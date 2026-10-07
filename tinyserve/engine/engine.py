"""Offline generation, one request at a time.

The engine keeps one paged pool for the calls it has seen. Full prompt
blocks can stay cached after a request finishes, so the next prompt that
starts with the same tokens skips that prefill. The pool is sized for the
longest request in a call. Phase 4 replaces the loop with continuous
batching over a shared pool.
"""

import time
from dataclasses import dataclass

import torch

from tinyserve.engine.model_runner import prepare_model_input, run_paged
from tinyserve.engine.sampler import sample_token
from tinyserve.engine.sequence import SamplingParams, Sequence, SequenceStatus
from tinyserve.kv.block_manager import BlockManager, blocks_for_tokens
from tinyserve.kv.cache import PagedKVCache
from tinyserve.kv.prefix_cache import PrefixCache
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


class Engine:
    """Run ``LlamaForCausalLM`` on a paged KV cache, one request at a time.

    ``generate`` is the spec's offline API. ``generate_tokens`` is the same
    loop for callers who already have ids (a chat template, or a test).
    ``block_size`` is the page size. The default matches ``CacheConfig``.
    Prefix caching is on by default for the same reason.
    """

    def __init__(
        self,
        model: LlamaForCausalLM,
        tokenizer: Tokenizer,
        block_size: int = 16,
        enable_prefix_caching: bool = True,
    ):
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self._next_seq_id = 0
        self._prefix: PrefixCache | None = None
        self._manager: BlockManager | None = None
        self._cache: PagedKVCache | None = None

    def generate(
        self, prompts: list[str], sampling_params: SamplingParams
    ) -> list[GenerationResult]:
        """Complete each prompt before starting the next.

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
        """Greedy or sampled completion for each token-id prompt, sequentially."""
        if not isinstance(sampling_params, SamplingParams):
            raise TypeError("sampling_params must be one SamplingParams for the whole call")
        if sampling_params.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if not prompts:
            return []
        # One pool for every prompt in the call. A later, longer call replaces
        # it, and the prefix cache starts empty again because the pages cannot grow.
        need = max(
            blocks_for_tokens(len(prompt_ids) + sampling_params.max_tokens, self.block_size)
            for prompt_ids in prompts
        )
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        self._ensure_pool(need, device, dtype)
        return [self._run_one(prompt_ids, sampling_params) for prompt_ids in prompts]

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

    def _reserve(self, sequence: Sequence, num_new_tokens: int) -> None:
        """Allocate `num_new_tokens`, evicting unused prefix blocks if the free list is short."""
        if self._manager is None:
            raise RuntimeError("the KV pool is missing")
        while not self._manager.can_allocate(sequence, num_new_tokens):
            if self._prefix is None:
                break
            evicted = self._prefix.evict_lru()
            if evicted is None:
                break
            self._manager.reclaim(evicted)
        self._manager.allocate(sequence, num_new_tokens)

    def _run_one(self, prompt_ids: list[int], params: SamplingParams) -> GenerationResult:
        """Prefill the uncached tail of `prompt_ids`, then sample up to `max_tokens`."""
        if not prompt_ids:
            raise ValueError("prompt is empty")
        limit = self.model.config.max_position_embeddings
        if len(prompt_ids) + params.max_tokens > limit:
            raise ValueError(
                f"prompt ({len(prompt_ids)}) plus max_tokens ({params.max_tokens}) "
                f"exceeds max_position_embeddings ({limit})"
            )
        if self._manager is None or self._cache is None:
            raise RuntimeError("the KV pool is missing")
        manager = self._manager
        cache = self._cache
        device = cache.k.device
        block_size = self.block_size
        sequence = Sequence(
            seq_id=self._next_seq_id,
            prompt_token_ids=list(prompt_ids),
            output_token_ids=[],
            sampling_params=params,
            status=SequenceStatus.RUNNING,
            num_computed_tokens=0,
            block_table=[],
            arrival_time=time.monotonic(),
            first_token_time=None,
        )
        self._next_seq_id += 1
        cached = self._take_cached_prefix(sequence, prompt_ids)
        remaining = len(prompt_ids) - cached
        self._reserve(sequence, remaining)
        generator = _request_generator(params, device)
        stop_ids = set(params.stop_token_ids)
        # perf_counter, and a GPU sync before each read: the forward returns
        # before the kernels finish, so an unsynced clock measures the launch.
        arrival = time.perf_counter()
        sequence.arrival_time = arrival

        prefill = prepare_model_input([sequence], [remaining], block_size, device)
        logits = run_paged(self.model, cache, prefill)  # [1, vocab]
        sequence.num_computed_tokens = len(prompt_ids)
        _sync(device)
        next_logits = logits[0]  # [vocab]
        token_times: list[float] = []
        for _ in range(params.max_tokens):
            token = int(sample_token(next_logits, params, generator).item())
            if token in stop_ids:
                break
            # `.item()` has already waited for the logits this token came from.
            # The forward below stores this token and builds the next logits.
            token_times.append(time.perf_counter())
            if sequence.first_token_time is None:
                sequence.first_token_time = token_times[0]
            sequence.output_token_ids.append(token)
            self._reserve(sequence, 1)
            step = prepare_model_input([sequence], [1], block_size, device)
            next_logits = run_paged(self.model, cache, step)[0]  # [vocab]
            sequence.num_computed_tokens += 1
            _sync(device)
        sequence.status = SequenceStatus.FINISHED
        if self._prefix is not None:
            self._prefix.cache_prompt(sequence.block_table, sequence.prompt_token_ids)
        manager.free(sequence)
        finished = time.perf_counter()
        ttft = None if not token_times else token_times[0] - arrival
        e2e = (token_times[-1] if token_times else finished) - arrival
        itl = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
        return GenerationResult(
            prompt_token_ids=sequence.prompt_token_ids,
            output_token_ids=list(sequence.output_token_ids),
            text=self.tokenizer.decode(sequence.output_token_ids),
            num_computed_tokens=sequence.num_computed_tokens,
            num_cached_prompt_tokens=cached,
            ttft_s=ttft,
            e2e_s=e2e,
            itl_s=itl,
        )


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
