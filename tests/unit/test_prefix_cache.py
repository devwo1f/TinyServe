"""Prefix hashes, the last-token rule, LRU eviction, and a reused prefill."""

from pathlib import Path

import torch

from tinyserve.engine.engine import Engine
from tinyserve.engine.sequence import SamplingParams, Sequence, SequenceStatus
from tinyserve.kv.block_manager import BlockManager
from tinyserve.kv.prefix_cache import PrefixCache, chain_hash
from tinyserve.model.llama import LlamaForCausalLM, LlamaModelConfig

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_llama.json"


def _seq() -> Sequence:
    return Sequence(
        seq_id=0,
        prompt_token_ids=[],
        output_token_ids=[],
        sampling_params=SamplingParams(),
        status=SequenceStatus.WAITING,
        num_computed_tokens=0,
        block_table=[],
        arrival_time=0.0,
        first_token_time=None,
    )


def test_a_later_block_misses_when_the_parent_prefix_differs():
    cache = PrefixCache(4)
    tokens = [10, 11, 12, 13, 20, 21, 22, 23, 30]
    cache.cache_prompt([0, 1], tokens)
    # Nine tokens leave the last one to recompute, so both full blocks can hit.
    assert cache.match(tokens) == [0, 1]
    # The second block's tokens match, but the first block does not.
    assert cache.match([99, 11, 12, 13, 20, 21, 22, 23, 30]) == []
    assert cache.match([10, 11, 12, 13, 20, 21, 22, 99, 30]) == [0]


def test_the_block_that_holds_the_last_prompt_token_is_not_reused():
    cache = PrefixCache(4)
    exact = [10, 11, 12, 13, 20, 21, 22, 23]
    cache.cache_prompt([0, 1], exact)
    # Both blocks are stored. A prompt of the same length still runs the last block.
    assert cache.match(exact) == [0]
    longer = exact + [30]
    assert cache.match(longer) == [0, 1]
    # The stored hash depends on the parent, not only on the tokens in the block.
    first = tuple(exact[:4])
    second = tuple(exact[4:])
    assert chain_hash(chain_hash(0, first), second) != chain_hash(0, second)


def test_lru_evicts_the_oldest_unused_block():
    cache = PrefixCache(4)
    manager = BlockManager(2, 4, is_cached=cache.is_cached, park_cached=cache.park)
    first = _seq()
    manager.allocate(first, 4)
    cache.cache_prompt(first.block_table, [1, 2, 3, 4])
    manager.free(first)
    second = _seq()
    manager.allocate(second, 4)
    cache.cache_prompt(second.block_table, [5, 6, 7, 8])
    manager.free(second)
    assert manager.num_free_blocks() == 0

    evicted = cache.evict_lru()
    assert evicted == 0
    manager.reclaim(0)
    assert cache.match([1, 2, 3, 4, 9]) == []
    assert cache.match([5, 6, 7, 8, 9]) == [1]
    # The parked block can be shared again. A free block cannot.
    reused = _seq()
    manager.share(reused, 1)
    cache.touch(1)
    assert reused.block_table == [1]
    assert manager.ref_count_of(1) == 1


def test_a_second_request_skips_the_shared_prefix_and_matches_a_cold_prefill():
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaModelConfig.from_json(FIXTURE)).float().eval()
    prompt = list(range(10))
    other = prompt[:8] + [40, 41]
    params = SamplingParams(temperature=0.0, max_tokens=3)
    warm = Engine(model, _Ascii(), block_size=4)
    cold = Engine(model, _Ascii(), block_size=4, enable_prefix_caching=False)
    # The first call has to finish before its blocks are cached. A single
    # batched call runs both prompts together, so the second would not see them.
    first = warm.generate_tokens([prompt], params)
    second = warm.generate_tokens([other], params)
    alone = cold.generate_tokens([other], params)[0]
    assert first[0].num_cached_prompt_tokens == 0
    assert second[0].num_cached_prompt_tokens == 8
    assert second[0].output_token_ids == alone.output_token_ids

    off = Engine(model, _Ascii(), block_size=4, enable_prefix_caching=False)
    uncached = off.generate_tokens([prompt, other], params)
    assert uncached[1].num_cached_prompt_tokens == 0
    assert uncached[1].output_token_ids == alone.output_token_ids


class _Ascii:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return "".join(chr(token) for token in token_ids)
