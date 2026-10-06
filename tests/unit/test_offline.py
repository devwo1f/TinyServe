"""Offline benchmark writes a Section 11 JSONL file. No committed timing numbers."""

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from bench.datasets import synthetic_workload
from bench.offline import run_offline
from tinyserve.engine.engine import Engine, GenerationResult
from tinyserve.model.llama import LlamaForCausalLM, LlamaModelConfig

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_llama.json"


class _Ascii:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return "".join(chr(token) for token in token_ids)


class _Delay:
    """Stand-in engine. Each call sleeps, so repeat ranking does not depend on the model."""

    def __init__(self, delays: list[float]):
        self.delays = delays
        self.calls = 0

        class _Weights:
            def parameters(self):
                yield torch.zeros(1)

        self.model = _Weights()

    def generate_tokens(self, prompts, params):
        time.sleep(self.delays[self.calls])
        self.calls += 1
        stops = set(params.stop_token_ids)
        output = []
        for token in range(params.max_tokens):
            if token in stops:
                break
            output.append(token)
        itl = [0.001] * (len(output) - 1)
        ttft = 0.002 if output else None
        e2e = 0.002 if not output else ttft + sum(itl)
        return [
            GenerationResult(
                prompt_token_ids=list(prompts[0]),
                output_token_ids=output,
                text="",
                num_computed_tokens=len(prompts[0]) + len(output),
                ttft_s=ttft,
                e2e_s=e2e,
                itl_s=itl,
            )
        ]


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_offline_jsonl_matches_its_own_clock(tmp_path):
    model = LlamaForCausalLM(LlamaModelConfig.from_json(FIXTURE)).float().eval()
    engine = Engine(model, _Ascii())
    samples = synthetic_workload(num_requests=3, input_len=4, output_len=2, seed=0, vocab_size=256)
    path = tmp_path / "offline.jsonl"
    summary = run_offline(
        engine,
        samples,
        path,
        config={"workload": "synthetic", "request_rate": float("inf")},
        num_warmup_requests=1,
        num_repeats=1,
        ignore_eos=True,
    )
    rows = _read(path)
    assert rows[0]["record"] == "meta"
    assert rows[0]["config"]["request_rate"] == "inf"
    assert rows[0]["git_commit"] == rows[0]["env_info"]["git_commit"]
    requests = [row for row in rows if row["record"] == "request"]
    assert [row["record"] for row in rows] == ["meta", "request", "request", "request", "summary"]
    assert rows[-1] == summary
    assert engine._next_seq_id == 4  # one warmup run, then all three samples
    for row in requests:
        assert row["prompt_len"] == 4
        assert row["num_output_tokens"] == 2
        assert row["ttft_s"] <= row["e2e_s"]
        assert len(row["itl_s"]) == 1
        assert row["tpot_s"] == pytest.approx(row["itl_s"][0])
        assert row["tpot_s"] == pytest.approx(row["e2e_s"] - row["ttft_s"])
    assert summary["num_requests"] == 3
    assert summary["num_warmup_requests"] == 1
    assert summary["output_tokens"] == 6
    assert summary["prompt_tokens"] == 12
    assert summary["output_throughput"] == pytest.approx(
        summary["output_tokens"] / summary["wall_clock_s"]
    )
    assert summary["total_throughput"] == pytest.approx(
        (summary["output_tokens"] + summary["prompt_tokens"]) / summary["wall_clock_s"]
    )
    assert summary["throughput_spread"] == 0.0
    ttft = [row["ttft_s"] for row in requests]
    assert summary["ttft_s"]["p50"] == pytest.approx(float(np.percentile(ttft, 50)))
    assert set(summary["ttft_s"]) == {"p50", "p90", "p99"}


def test_median_repeat_is_the_middle_throughput(tmp_path):
    samples = synthetic_workload(num_requests=1, input_len=4, output_len=2, seed=1, vocab_size=256)
    # One sample, no warmup, three repeats. Sleeps dominate the wall clock.
    engine = _Delay([0.08, 0.01, 0.25])
    summary = run_offline(
        engine,
        samples,
        tmp_path / "repeats.jsonl",
        config={"workload": "synthetic"},
        num_warmup_requests=0,
        num_repeats=3,
    )
    assert engine.calls == 3
    assert summary["median_repeat"] == 0
    assert summary["repeats"][0]["wall_clock_s"] == summary["wall_clock_s"]
    assert summary["throughput_spread"] == pytest.approx(
        summary["repeats"][1]["output_throughput"] - summary["repeats"][2]["output_throughput"]
    )
    rows = _read(tmp_path / "repeats.jsonl")
    assert sum(row["record"] == "request" for row in rows) == 3
    assert all(row["repeat"] in (0, 1, 2) for row in rows if row["record"] == "request")


def test_warmup_runs_are_omitted_and_eos_can_stop(tmp_path):
    samples = synthetic_workload(num_requests=2, input_len=3, output_len=4, seed=2, vocab_size=256)
    engine = _Delay([0.0] * 6)  # one warmup + two measured, twice
    summary = run_offline(
        engine,
        samples,
        tmp_path / "warm.jsonl",
        config={},
        num_warmup_requests=1,
        num_repeats=2,
        ignore_eos=False,
        eos_token_id=1,
    )
    assert engine.calls == 6
    assert summary["num_requests"] == 2
    assert summary["num_warmup_requests"] == 1
    rows = [row for row in _read(tmp_path / "warm.jsonl") if row["record"] == "request"]
    assert len(rows) == 4
    # output ids are range(max_tokens) until eos 1, so only id 0 is kept.
    assert all(row["num_output_tokens"] == 1 for row in rows)
    assert all(row["tpot_s"] is None for row in rows)
    assert summary["tpot_s"] == {"p50": None, "p90": None, "p99": None}


def test_offline_rejects_an_empty_set(tmp_path):
    engine = _Delay([])
    with pytest.raises(ValueError, match="empty"):
        run_offline(engine, [], tmp_path / "no.jsonl", config={})
    samples = synthetic_workload(num_requests=1, input_len=2, output_len=1, seed=0, vocab_size=8)
    with pytest.raises(ValueError, match="eos_token_id"):
        run_offline(
            engine,
            samples,
            tmp_path / "no.jsonl",
            config={},
            ignore_eos=False,
        )
