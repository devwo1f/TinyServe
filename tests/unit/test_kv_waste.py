"""Slot waste: the last-block gap, and a dense cache that reserves max_len."""

from pathlib import Path

import pytest

from tinyserve.engine.engine import Engine
from tinyserve.engine.sequence import SamplingParams, Sequence, SequenceStatus
from tinyserve.kv.block_manager import BlockManager
from tinyserve.kv.waste import (
    concurrent_slot_sample,
    kv_memory_waste,
    request_step_samples,
    sample_slots,
)
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


class _Ids:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return ""


def test_waste_is_the_unused_fraction_of_the_allocation():
    assert kv_memory_waste(32, 17) == pytest.approx(15 / 32)
    assert kv_memory_waste(0, 0) == 0.0
    with pytest.raises(ValueError, match="exceed"):
        kv_memory_waste(4, 5)


def test_a_partial_last_block_is_paged_waste_and_max_len_is_contiguous():
    # 15 and 16 tokens fill one block of 16. 17 tokens open a second block.
    for used, allocated in ((15, 16), (16, 16), (17, 32)):
        manager = BlockManager(4, 16)
        sequence = _seq()
        manager.allocate(sequence, used)
        sequence.num_computed_tokens = used
        sample = sample_slots([sequence.block_table], [used], block_size=16, max_len=64)
        assert sample.paged_allocated_slots == allocated
        assert sample.used_slots == used
        assert sample.contiguous_allocated_slots == 64
        assert sample.paged_waste == pytest.approx((allocated - used) / allocated)
        assert sample.contiguous_waste == pytest.approx((64 - used) / 64)


def test_a_short_block_table_is_rejected():
    with pytest.raises(ValueError, match="does not cover"):
        sample_slots([[0]], [5], block_size=4, max_len=16)


def test_concurrent_sequences_share_the_max_length_reservation():
    # Lengths 6 and 4 with block size 4 need 8 + 4 paged slots. Both pay max_len.
    sample = concurrent_slot_sample([6, 4], block_size=4, max_len=16)
    assert sample.used_slots == 10
    assert sample.paged_allocated_slots == 12
    assert sample.contiguous_allocated_slots == 32
    assert sample.paged_waste == pytest.approx(2 / 12)
    assert sample.contiguous_waste == pytest.approx(22 / 32)


def test_engine_steps_match_the_block_manager_walk():
    model = LlamaForCausalLM(LlamaModelConfig.from_json(FIXTURE)).float().eval()
    engine = Engine(
        model,
        _Ids(),
        block_size=4,
        enable_prefix_caching=False,
        record_kv_waste=True,
        contiguous_max_len=64,
    )
    result = engine.generate_tokens(
        [[1, 2, 3, 4, 5]], SamplingParams(temperature=0.0, max_tokens=2)
    )
    expected = request_step_samples(5, 2, block_size=4, max_len=64)
    assert result[0].slot_samples == tuple(expected)
    assert [sample.used_slots for sample in result[0].slot_samples] == [5, 6, 7]
    assert [sample.paged_allocated_slots for sample in result[0].slot_samples] == [8, 8, 8]
    assert all(sample.contiguous_allocated_slots == 64 for sample in result[0].slot_samples)


def test_recording_is_off_unless_asked():
    model = LlamaForCausalLM(LlamaModelConfig.from_json(FIXTURE)).float().eval()
    engine = Engine(model, _Ids(), block_size=4, enable_prefix_caching=False)
    result = engine.generate_tokens([[1, 2, 3]], SamplingParams(temperature=0.0, max_tokens=1))
    assert result[0].slot_samples == ()
