"""vLLM result rows, without importing vLLM. The 1B run is the separate virtualenv."""

import json
from types import SimpleNamespace

import pytest

from bench.datasets import synthetic_workload
from bench.vllm_offline import latency_seconds, rows_for_repeat


def test_latency_adds_the_monotonic_gap_to_wall_clock_ttft():
    metrics = SimpleNamespace(first_token_latency=0.25, first_token_ts=100.0, last_token_ts=100.75)
    ttft, e2e = latency_seconds(metrics)
    assert ttft == pytest.approx(0.25)
    assert e2e == pytest.approx(1.0)


def test_latency_rejects_missing_timestamps():
    metrics = SimpleNamespace(first_token_latency=0.0, first_token_ts=0.0, last_token_ts=0.0)
    with pytest.raises(RuntimeError, match="missing"):
        latency_seconds(metrics)


def test_rows_keep_fixed_lengths_and_leave_itl_empty():
    samples = synthetic_workload(num_requests=2, input_len=4, output_len=3, seed=0, vocab_size=32)
    rows = rows_for_repeat(samples, 0, [(3, 0.1, 0.4), (3, 0.2, 0.5)])
    assert [row["num_output_tokens"] for row in rows] == [3, 3]
    assert all(row["itl_s"] == [] for row in rows)
    assert rows[0]["tpot_s"] == pytest.approx((0.4 - 0.1) / 2)
    json.dumps(rows)


def test_rows_reject_a_short_generation():
    samples = synthetic_workload(num_requests=1, input_len=2, output_len=4, seed=1, vocab_size=16)
    with pytest.raises(RuntimeError, match="emitted 2"):
        rows_for_repeat(samples, 0, [(2, 0.1, 0.2)])
