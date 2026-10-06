"""Offline throughput: run a fixed request set back to back and record latency.

There is no arrival process here. `request_rate = inf` in the benchmark config
means the same thing: every request is started as soon as the previous one
finishes. Poisson arrivals belong to the serving benchmark.

Warmup is a separate pass over the first few samples. Those runs are not
written into the result file, and the measured pass still includes every
sample, so `num_requests` stays the count that was asked for.

The result file is JSONL, one object per line (spec Section 11): a meta line,
one line per measured request per repeat, and a summary line. The summary's
percentiles come from the median repeat, ranked by output throughput. The
spread is the gap between the fastest and slowest repeat.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from bench.datasets import Sample
from scripts.env_info import collect_env_info
from tinyserve.engine.engine import Engine
from tinyserve.engine.sequence import SamplingParams


def run_offline(
    engine: Engine,
    samples: list[Sample],
    output_path: str | Path,
    *,
    config: dict,
    num_warmup_requests: int = 0,
    num_repeats: int = 1,
    ignore_eos: bool = True,
    eos_token_id: int | None = None,
) -> dict:
    """Run `samples` through `engine` and write the Section 11 JSONL file.

    Returns the summary object that was written as the last line.
    """
    if not samples:
        raise ValueError("samples is empty")
    if num_warmup_requests < 0:
        raise ValueError("num_warmup_requests must be >= 0")
    if num_repeats < 1:
        raise ValueError("num_repeats must be at least 1")
    if not ignore_eos and eos_token_id is None:
        raise ValueError("eos_token_id is required when ignore_eos is False")

    device = next(engine.model.parameters()).device
    warmup = samples[:num_warmup_requests]
    request_rows: list[dict] = []
    repeat_rows: list[dict] = []
    for repeat in range(num_repeats):
        for sample in warmup:
            _generate_one(engine, sample, ignore_eos=ignore_eos, eos_token_id=eos_token_id)
        # The clock starts after warmup so those forwards are not in the throughput.
        wall_start = _wall_start(device)
        measured: list[dict] = []
        for index, sample in enumerate(samples):
            result = _generate_one(engine, sample, ignore_eos=ignore_eos, eos_token_id=eos_token_id)
            n_out = len(result.output_token_ids)
            itl = list(result.itl_s or [])
            tpot = _tpot(result.e2e_s, result.ttft_s, n_out)
            measured.append(
                {
                    "record": "request",
                    "repeat": repeat,
                    "index": index,
                    "workload": sample.workload,
                    "prompt_len": len(sample.prompt_token_ids),
                    "num_output_tokens": n_out,
                    "ttft_s": result.ttft_s,
                    "e2e_s": result.e2e_s,
                    "tpot_s": tpot,
                    "itl_s": itl,
                }
            )
        wall_s = _wall_end(device, wall_start)
        output_tokens = sum(row["num_output_tokens"] for row in measured)
        prompt_tokens = sum(row["prompt_len"] for row in measured)
        repeat_rows.append(
            {
                "repeat": repeat,
                "wall_clock_s": wall_s,
                "output_tokens": output_tokens,
                "prompt_tokens": prompt_tokens,
                "output_throughput": output_tokens / wall_s,
                "total_throughput": (output_tokens + prompt_tokens) / wall_s,
            }
        )
        request_rows.extend(measured)

    summary = _summary(repeat_rows, request_rows, num_warmup_requests)
    env = collect_env_info()
    meta = {
        "record": "meta",
        "config": config,
        "env_info": env,
        "git_commit": env.get("git_commit"),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [meta, *request_rows, summary]
    path.write_text("".join(json.dumps(_json_ready(line)) + "\n" for line in lines))
    return summary


def _generate_one(engine: Engine, sample: Sample, *, ignore_eos: bool, eos_token_id: int | None):
    """One sample, with max_tokens fixed to the dataset length."""
    stops: list[int] = []
    if not ignore_eos:
        assert eos_token_id is not None
        stops = [eos_token_id]
    params = SamplingParams(
        temperature=0.0,
        max_tokens=sample.output_len,
        stop_token_ids=stops,
    )
    return engine.generate_tokens([sample.prompt_token_ids], params)[0]


def _tpot(e2e_s: float, ttft_s: float | None, n_out: int) -> float | None:
    """Mean time per output token after the first one. Undefined for a single token."""
    if n_out < 2 or ttft_s is None:
        return None
    return (e2e_s - ttft_s) / (n_out - 1)


def _wall_start(device: torch.device) -> float:
    _sync(device)
    return time.perf_counter()


def _wall_end(device: torch.device, start: float) -> float:
    _sync(device)
    return time.perf_counter() - start


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(repeat_rows: list[dict], request_rows: list[dict], num_warmup_requests: int) -> dict:
    """Percentiles from the median repeat. Spread is across repeats."""
    ranked = sorted(repeat_rows, key=lambda row: row["output_throughput"])
    median = ranked[len(ranked) // 2]
    median_index = median["repeat"]
    chosen = [row for row in request_rows if row["repeat"] == median_index]
    throughputs = [row["output_throughput"] for row in repeat_rows]
    return {
        "record": "summary",
        "num_warmup_requests": num_warmup_requests,
        "num_requests": len(chosen),
        "num_repeats": len(repeat_rows),
        "median_repeat": median_index,
        "wall_clock_s": median["wall_clock_s"],
        "output_tokens": median["output_tokens"],
        "prompt_tokens": median["prompt_tokens"],
        "output_throughput": median["output_throughput"],
        "total_throughput": median["total_throughput"],
        "throughput_spread": max(throughputs) - min(throughputs),
        "repeats": repeat_rows,
        "ttft_s": _percentiles([row["ttft_s"] for row in chosen if row["ttft_s"] is not None]),
        "e2e_s": _percentiles([row["e2e_s"] for row in chosen]),
        "tpot_s": _percentiles([row["tpot_s"] for row in chosen if row["tpot_s"] is not None]),
        "itl_s": _percentiles([dt for row in chosen for dt in row["itl_s"]]),
    }


def _json_ready(value):
    """Replace non-finite floats so a config with `request_rate=inf` still writes."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _percentiles(values: list[float]) -> dict:
    """p50, p90, and p99. Empty input stays null instead of inventing a zero."""
    if not values:
        return {"p50": None, "p90": None, "p99": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
    }
