"""Paged KV sizing with fake memory numbers. No GPU."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tinyserve.config import CacheConfig
from tinyserve.kv.cache import (
    PagedKVCache,
    build_paged_cache,
    bytes_per_token,
    cache_profile,
    kv_budget_bytes,
    measure_activation_bytes,
    num_blocks_for_budget,
    weight_bytes,
)
from tinyserve.model.llama import LlamaForCausalLM, LlamaModelConfig

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_llama.json"


def test_spec_8b_token_is_128_kib():
    # 32 layers * 8 KV heads * 128 * 2 (K and V) * 2 bytes
    assert bytes_per_token(32, 8, 128, torch.bfloat16) == 131_072
    assert (1024**3) // 131_072 == 8192


def test_block_count_floors_after_the_safety_margin():
    budget = kv_budget_bytes(
        total_gpu_bytes=1_000_000,
        gpu_memory_utilization=0.9,
        weight_bytes=100_000,
        activation_bytes=50_000,
        safety_margin_bytes=10_000,
    )
    assert budget == 740_000
    assert num_blocks_for_budget(budget, 8192) == 90
    profile = cache_profile(
        num_layers=2,
        num_kv_heads=2,
        head_dim=16,
        dtype=torch.float32,
        block_size=16,
        total_gpu_bytes=1_000_000,
        gpu_memory_utilization=0.9,
        weight_bytes=100_000,
        activation_bytes=50_000,
        safety_margin_bytes=10_000,
    )
    # 2 * 2 * 16 * 2 * 4 = 512 bytes/token, * 16 = 8192 bytes/block
    assert profile.bytes_per_block == 8192
    assert profile.num_blocks == 90
    assert profile.capacity_tokens == 1440
    assert str(profile) == "KV cache: 90 blocks, 1440 tokens (8192 bytes/block)"


def test_budget_below_one_block_is_zero_blocks():
    assert num_blocks_for_budget(100, 8192) == 0
    assert num_blocks_for_budget(-1, 8192) == 0


def test_override_ignores_the_budget():
    profile = cache_profile(
        num_layers=2,
        num_kv_heads=2,
        head_dim=16,
        dtype=torch.float32,
        block_size=16,
        total_gpu_bytes=1_000,
        gpu_memory_utilization=0.5,
        weight_bytes=900,
        activation_bytes=0,
        safety_margin_bytes=0,
        num_blocks_override=4,
    )
    assert profile.used_override
    assert profile.num_blocks == 4
    assert profile.capacity_tokens == 64
    assert profile.kv_budget_bytes < profile.bytes_per_block


def test_tied_embeddings_are_counted_once():
    config = LlamaModelConfig.from_json(FIXTURE)
    untied = LlamaForCausalLM(replace(config, tie_word_embeddings=False))
    tied = LlamaForCausalLM(replace(config, tie_word_embeddings=True))
    extra = config.vocab_size * config.hidden_size * 4  # float32 LM head
    assert weight_bytes(untied) - weight_bytes(tied) == extra


def test_paged_cache_shape_matches_the_spec():
    cache = PagedKVCache(
        num_blocks=3,
        block_size=16,
        num_layers=2,
        num_kv_heads=2,
        head_dim=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert cache.capacity_tokens == 48
    k, v = cache.layer_kv(1)
    assert k.shape == (3, 16, 2, 16)  # [num_blocks, block_size, num_kv_heads, head_dim]
    assert v.shape == k.shape
    assert torch.count_nonzero(cache.k) == 0


def test_build_uses_fake_activation_bytes_on_cpu():
    model = LlamaForCausalLM(LlamaModelConfig.from_json(FIXTURE)).eval()
    config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        memory_safety_margin_gib=0,
    )
    cache, profile = build_paged_cache(
        model,
        config,
        total_gpu_bytes=20_000_000,
        max_num_batched_tokens=4,
        activation_bytes=1_000,
    )
    assert profile.activation_bytes == 1_000
    assert profile.num_blocks == cache.num_blocks
    assert profile.capacity_tokens == cache.capacity_tokens
    assert cache.k.shape[0] == model.config.num_hidden_layers
    assert "blocks" in str(profile) and "tokens" in str(profile)


def test_build_raises_when_the_budget_cannot_hold_a_block():
    model = LlamaForCausalLM(LlamaModelConfig.from_json(FIXTURE)).eval()
    config = CacheConfig(memory_safety_margin_gib=0)
    with pytest.raises(RuntimeError, match="below one block"):
        build_paged_cache(
            model,
            config,
            total_gpu_bytes=1_000,
            max_num_batched_tokens=1,
            activation_bytes=0,
        )


def test_cpu_activation_profiling_refuses_to_return_zero():
    with pytest.raises(RuntimeError, match="needs CUDA"):
        measure_activation_bytes(lambda: None, torch.device("cpu"))
