"""Arrival loop and plots. Timing numbers here are not benchmark results."""

import json
from pathlib import Path

from bench.batching import poisson_offsets, run_arrivals, run_concurrent
from bench.datasets import synthetic_workload
from bench.plots import plot_directory
from tinyserve.engine.engine import Engine
from tinyserve.engine.sequence import SamplingParams
from tinyserve.model.llama import LlamaForCausalLM, LlamaModelConfig

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_llama.json"


class _Ascii:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return "".join(chr(token) for token in token_ids)


def _model():
    return LlamaForCausalLM(LlamaModelConfig.from_json(FIXTURE)).float().eval()


def test_poisson_offsets_start_at_zero_and_repeat_for_a_seed():
    first = poisson_offsets(4, 2.0, seed=7)
    again = poisson_offsets(4, 2.0, seed=7)
    assert first == again
    assert first[0] == 0.0
    assert first == sorted(first)
    assert len(first) == 4
    assert poisson_offsets(4, 2.0, seed=8) != first


def test_a_delayed_arrival_matches_an_offline_batch():
    model = _model()
    prompts = [[1, 2, 3, 4], [8, 9, 10]]
    params = SamplingParams(temperature=0.0, max_tokens=3)
    expected = Engine(model, _Ascii(), enable_prefix_caching=False).generate_tokens(prompts, params)
    staggered = Engine(model, _Ascii(), enable_prefix_caching=False)
    got = run_arrivals(staggered, prompts, params, [0.0, 0.02])
    assert [row.output_token_ids for row in got] == [row.output_token_ids for row in expected]


def test_concurrent_jsonl_counts_every_output_token(tmp_path):
    model = _model()
    engine = Engine(model, _Ascii(), enable_prefix_caching=False)
    samples = synthetic_workload(num_requests=2, input_len=4, output_len=2, seed=1, vocab_size=256)
    path = tmp_path / "batched.jsonl"
    summary = run_concurrent(
        engine,
        samples,
        path,
        config={"workload": "synthetic", "series": "offline", "label": "batched"},
        num_warmup_requests=0,
        num_repeats=1,
    )
    assert summary["num_requests"] == 2
    assert summary["output_tokens"] == 4
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["record"] for row in rows] == ["meta", "request", "request", "summary"]
    assert all(row["itl_s"] for row in rows if row["record"] == "request")


def test_plot_directory_writes_a_png_from_result_files(tmp_path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    _write(
        result_dir / "naive.jsonl",
        {"series": "offline", "label": "naive"},
        output_throughput=10.0,
    )
    _write(
        result_dir / "batched.jsonl",
        {"series": "offline", "label": "batched"},
        output_throughput=30.0,
    )
    _write(
        result_dir / "poisson.jsonl",
        {"series": "poisson", "label": "4", "request_rate": 4.0},
        output_throughput=20.0,
        itl_p99=0.05,
    )
    _write(
        result_dir / "chunk.jsonl",
        {"series": "chunk", "label": "32", "max_num_batched_tokens": 32},
        output_throughput=15.0,
        itl_p99=0.02,
    )
    written = plot_directory(result_dir, tmp_path / "plots")
    names = sorted(path.name for path in written)
    assert names == ["p4-4-chunk-itl.png", "p4-4-offline-throughput.png", "p4-4-poisson.png"]
    assert all(path.stat().st_size > 0 for path in written)


def _write(path: Path, config: dict, *, output_throughput: float, itl_p99: float = 0.01) -> None:
    meta = {"record": "meta", "config": config}
    summary = {
        "record": "summary",
        "output_throughput": output_throughput,
        "itl_s": {"p50": itl_p99, "p90": itl_p99, "p99": itl_p99},
    }
    path.write_text(json.dumps(meta) + "\n" + json.dumps(summary) + "\n")
