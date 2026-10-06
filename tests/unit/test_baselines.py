"""Baseline helpers on the tiny model. The 1B numbers come from the script, not from here."""

import json
from pathlib import Path

import pytest
from transformers import LlamaConfig as HFConfig
from transformers import LlamaForCausalLM as HFLlama

from bench.baselines import profile_generation, run_hf_generate
from bench.datasets import synthetic_workload
from tinyserve.engine.engine import Engine
from tinyserve.model.llama import LlamaForCausalLM, LlamaModelConfig

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_llama.json"


class _Ascii:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return "".join(chr(token) for token in token_ids)


def _hf_tiny() -> HFLlama:
    raw = json.loads(FIXTURE.read_text())
    config = HFConfig(
        vocab_size=raw["vocab_size"],
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        max_position_embeddings=raw["max_position_embeddings"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_theta"],
        rope_scaling=raw["rope_scaling"],
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
        attn_implementation="eager",
        eos_token_id=raw["eos_token_id"],
    )
    return HFLlama(config).float().eval()


def test_hf_generate_writes_the_same_jsonl_shape(tmp_path):
    samples = synthetic_workload(num_requests=2, input_len=4, output_len=3, seed=0, vocab_size=256)
    path = tmp_path / "hf.jsonl"
    summary = run_hf_generate(
        _hf_tiny(),
        samples,
        path,
        config={"engine": "hf_generate", "request_rate": float("inf")},
        num_warmup_requests=0,
        num_repeats=1,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["record"] for row in rows] == ["meta", "request", "request", "summary"]
    assert rows[-1] == summary
    for row in rows[1:3]:
        assert row["num_output_tokens"] == 3
        assert row["prompt_len"] == 4
        assert row["ttft_s"] <= row["e2e_s"]
        assert len(row["itl_s"]) == 2
        assert row["tpot_s"] == pytest.approx((row["e2e_s"] - row["ttft_s"]) / 2)
    assert summary["output_tokens"] == 6
    assert summary["output_throughput"] == pytest.approx(
        summary["output_tokens"] / summary["wall_clock_s"]
    )


def test_profile_summary_is_script_shaped(tmp_path):
    model = LlamaForCausalLM(LlamaModelConfig.from_json(FIXTURE)).float().eval()
    engine = Engine(model, _Ascii())
    sample = synthetic_workload(num_requests=1, input_len=4, output_len=2, seed=1, vocab_size=256)[
        0
    ]
    body = profile_generation(engine, sample)
    encoded = json.dumps(body)
    assert "table" in encoded
    assert isinstance(body["table"], str)
    assert body["table"]
    assert body["top_self_time"]
    assert set(body["groups"]["us"]) == {"attention", "gemm", "norm", "elementwise", "other"}
    assert body["prompt_len"] == 4
    assert body["output_len"] == 2
    json.loads(encoded)
    assert "CPU" in body["activities"]
