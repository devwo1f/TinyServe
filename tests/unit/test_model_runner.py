"""Flattened paged forwards match the contiguous cache, including a mixed batch."""

from pathlib import Path

import pytest
import torch

from tinyserve.engine.model_runner import prepare_model_input, run_paged
from tinyserve.engine.sequence import SamplingParams, Sequence, SequenceStatus
from tinyserve.kv.block_manager import BlockManager
from tinyserve.kv.cache import PagedKVCache
from tinyserve.model.llama import LlamaForCausalLM, LlamaModelConfig

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_llama.json"


def _model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaModelConfig.from_json(FIXTURE)).float().eval()


def _sequence(prompt: list[int], output: list[int] | None = None, computed: int = 0) -> Sequence:
    return Sequence(
        seq_id=0,
        prompt_token_ids=prompt,
        output_token_ids=list(output or []),
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
        status=SequenceStatus.RUNNING,
        num_computed_tokens=computed,
        block_table=[],
        arrival_time=0.0,
        first_token_time=None,
    )


def _cache(model: LlamaForCausalLM, num_blocks: int, block_size: int) -> PagedKVCache:
    config = model.config
    cache = PagedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    # Unused offsets must not be read. A zero fill would hide that.
    cache.k.fill_(3)
    cache.v.fill_(-3)
    return cache


def test_prepare_packs_slots_and_pads_the_block_table_with_minus_one():
    seq_a = _sequence([10, 11, 12, 13, 14])
    seq_a.block_table = [3, 6]
    seq_b = _sequence([1, 2, 3], output=[4], computed=3)
    seq_b.block_table = [1]
    prepared = prepare_model_input([seq_a, seq_b], [5, 1], block_size=4, device=torch.device("cpu"))

    assert prepared.input_ids.tolist() == [10, 11, 12, 13, 14, 4]
    assert prepared.positions.tolist() == [0, 1, 2, 3, 4, 3]
    # p 0..3 -> block 3; p 4 -> block 6 offset 0; seq B p 3 -> block 1 offset 3
    assert prepared.slot_mapping.tolist() == [12, 13, 14, 15, 24, 7]
    assert prepared.query_start_loc.tolist() == [0, 5, 6]
    assert prepared.seq_lens.tolist() == [5, 4]
    assert prepared.logits_indices.tolist() == [4, 5]
    assert prepared.block_tables.tolist() == [[3, 6], [1, -1]]
    assert seq_a.num_computed_tokens == 0


def test_paged_prefill_and_a_later_chunk_match_the_full_forward():
    model = _model()
    ids = torch.randint(0, 256, (7,)).tolist()
    with torch.no_grad():
        full = model(torch.tensor([ids]))  # [1, 7, vocab]

    block_size = 4
    device = torch.device("cpu")
    cache = _cache(model, num_blocks=4, block_size=block_size)
    manager = BlockManager(4, block_size)
    whole = _sequence(ids)
    manager.allocate(whole, 7)
    with torch.no_grad():
        prefill = prepare_model_input([whole], [7], block_size, device)
        prefill.logits_indices = torch.arange(7)
        prefill_logits = run_paged(model, cache, prefill)
    # Same query length as the contiguous forward, so the SDPA call matches.
    assert torch.equal(prefill_logits, full[0])

    cache = _cache(model, num_blocks=4, block_size=block_size)
    manager = BlockManager(4, block_size)
    seq = _sequence(ids)
    manager.allocate(seq, 3)
    with torch.no_grad():
        prefix = prepare_model_input([seq], [3], block_size, device)
        prefix.logits_indices = torch.arange(3)
        prefix_logits = run_paged(model, cache, prefix)
        seq.num_computed_tokens = 3
        # A 3-token causal kernel and the prefix of a 7-token one differ in the last bits.
        torch.testing.assert_close(prefix_logits, full[0, :3], atol=1e-5, rtol=0)

        manager.allocate(seq, 4)
        chunk = prepare_model_input([seq], [4], block_size, device)
        chunk.logits_indices = torch.arange(4)
        chunk_logits = run_paged(model, cache, chunk)
    torch.testing.assert_close(chunk_logits, full[0, 3:], atol=1e-5, rtol=0)


def test_one_step_can_prefill_one_sequence_and_decode_another():
    model = _model()
    ids_a = torch.randint(0, 256, (5,)).tolist()
    ids_b = torch.randint(0, 256, (5,)).tolist()
    with torch.no_grad():
        full_a = model(torch.tensor([ids_a]))
        full_b = model(torch.tensor([ids_b]))

    block_size = 4
    cache = _cache(model, num_blocks=6, block_size=block_size)
    manager = BlockManager(6, block_size)
    seq_b = _sequence(ids_b)
    manager.allocate(seq_b, 4)
    with torch.no_grad():
        warm = prepare_model_input([seq_b], [4], block_size, torch.device("cpu"))
        run_paged(model, cache, warm)
        seq_b.num_computed_tokens = 4

        seq_a = _sequence(ids_a)
        manager.allocate(seq_a, 5)
        manager.allocate(seq_b, 1)
        prepared = prepare_model_input([seq_a, seq_b], [5, 1], block_size, torch.device("cpu"))
        logits = run_paged(model, cache, prepared)
    torch.testing.assert_close(logits[0], full_a[0, -1], atol=1e-5, rtol=0)
    torch.testing.assert_close(logits[1], full_b[0, -1], atol=1e-5, rtol=0)


def test_a_position_past_the_block_table_is_rejected():
    seq = _sequence([1, 2, 3, 4, 5])
    seq.block_table = [0]
    with pytest.raises(ValueError, match="block table"):
        prepare_model_input([seq], [5], block_size=4, device=torch.device("cpu"))
