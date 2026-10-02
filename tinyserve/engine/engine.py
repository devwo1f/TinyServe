"""Offline generation, one request at a time.

The contiguous cache is one dense buffer per sequence. A second request would
need its own buffer or a scheduler, and neither exists yet, so this loop
finishes a prompt before starting the next. Phase 4 replaces the loop with
continuous batching; the sampler and the model stay.
"""

import time
from dataclasses import dataclass

import torch

from tinyserve.engine.sampler import sample_token
from tinyserve.engine.sequence import SamplingParams, Sequence, SequenceStatus
from tinyserve.model.llama import ContiguousKVCache, LlamaForCausalLM
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


class Engine:
    """Run `LlamaForCausalLM` with the contiguous cache and the sampler.

    `generate` is the spec's offline API. `generate_tokens` is the same loop
    for callers who already have ids (a chat template, or a test).
    """

    def __init__(self, model: LlamaForCausalLM, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._next_seq_id = 0

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
        return [self._run_one(prompt_ids, sampling_params) for prompt_ids in prompts]

    def _run_one(self, prompt_ids: list[int], params: SamplingParams) -> GenerationResult:
        """Prefill `prompt_ids`, then sample up to `max_tokens` new ids into a fresh cache."""
        if not prompt_ids:
            raise ValueError("prompt is empty")
        limit = self.model.config.max_position_embeddings
        if len(prompt_ids) + params.max_tokens > limit:
            raise ValueError(
                f"prompt ({len(prompt_ids)}) plus max_tokens ({params.max_tokens}) "
                f"exceeds max_position_embeddings ({limit})"
            )
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        # One buffer for this request only. The next request allocates its own.
        cache = ContiguousKVCache(
            self.model.config,
            batch=1,
            max_len=len(prompt_ids) + params.max_tokens,
            dtype=dtype,
            device=device,
        )
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
        generator = _request_generator(params, device)
        stop_ids = set(params.stop_token_ids)

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # [1, S]
        logits = self.model(input_ids, cache=cache)  # [1, S, vocab]
        next_logits = logits[0, -1]  # [vocab]
        for _ in range(params.max_tokens):
            token = int(sample_token(next_logits, params, generator).item())
            if token in stop_ids:
                break
            if sequence.first_token_time is None:
                sequence.first_token_time = time.monotonic()
            sequence.output_token_ids.append(token)
            step = torch.tensor([[token]], dtype=torch.long, device=device)  # [1, 1]
            # Writing this token's KV keeps num_computed_tokens equal to the
            # tokens that are actually in the cache, including the last one.
            next_logits = self.model(step, cache=cache)[0, -1]  # [vocab]
        sequence.status = SequenceStatus.FINISHED
        sequence.num_computed_tokens = cache.length
        return GenerationResult(
            prompt_token_ids=sequence.prompt_token_ids,
            output_token_ids=list(sequence.output_token_ids),
            text=self.tokenizer.decode(sequence.output_token_ids),
            num_computed_tokens=sequence.num_computed_tokens,
        )


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
