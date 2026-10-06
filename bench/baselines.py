"""Naive TinyServe and Hugging Face ``generate`` baselines on the same requests.

The two runs are separate processes of work, not one process with both models
loaded: a 1B checkpoint in bf16 does not leave room for a second copy on an
8 GB GPU. Both write the Section 11 JSONL from ``bench.offline``. A third file
is a ``torch.profiler`` summary of one TinyServe request, so the learning note
can say where the time went without a hand-written table.

Hugging Face ``generate`` is the baseline, not a hand-rolled argmax loop. The
shipped Llama generation config samples, so this call forces greedy decoding
and sets the minimum new tokens equal to the maximum. That is the ``ignore_eos``
equivalent: both engines emit the dataset length.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers.generation.streamers import BaseStreamer

from bench.datasets import Sample, synthetic_workload
from bench.offline import (
    _json_ready,
    _wall_end,
    _wall_start,
    repeat_record,
    request_record,
    run_offline,
    write_benchmark_jsonl,
)
from scripts.env_info import collect_env_info
from tinyserve.engine.engine import Engine
from tinyserve.engine.sequence import SamplingParams


def run_hf_generate(
    model,
    samples: list[Sample],
    output_path: str | Path,
    *,
    config: dict,
    num_warmup_requests: int = 0,
    num_repeats: int = 1,
) -> dict:
    """Time ``model.generate`` on each sample and write the Section 11 file.

    One request at a time, same warmup and repeat rules as ``run_offline``.
    """
    if not samples:
        raise ValueError("samples is empty")
    if num_warmup_requests < 0:
        raise ValueError("num_warmup_requests must be >= 0")
    if num_repeats < 1:
        raise ValueError("num_repeats must be at least 1")

    device = next(model.parameters()).device
    warmup = samples[:num_warmup_requests]
    request_rows: list[dict] = []
    repeat_rows: list[dict] = []
    for repeat in range(num_repeats):
        for sample in warmup:
            _generate_hf(model, sample, device)
        wall_start = _wall_start(device)
        measured: list[dict] = []
        for index, sample in enumerate(samples):
            n_out, ttft_s, e2e_s, itl_s = _generate_hf(model, sample, device)
            measured.append(
                request_record(
                    repeat,
                    index,
                    sample,
                    num_output_tokens=n_out,
                    ttft_s=ttft_s,
                    e2e_s=e2e_s,
                    itl_s=itl_s,
                )
            )
        repeat_rows.append(repeat_record(repeat, measured, _wall_end(device, wall_start)))
        request_rows.extend(measured)
    return write_benchmark_jsonl(
        output_path,
        config=config,
        request_rows=request_rows,
        repeat_rows=repeat_rows,
        num_warmup_requests=num_warmup_requests,
    )


def profile_generation(engine: Engine, sample: Sample) -> dict:
    """Profile one TinyServe request. Self time, so parent ops are not added twice.

    The chrome trace is not written. It is large, and the summary table is what
    the learning note needs.
    """
    from torch.profiler import ProfilerActivity, profile

    device = next(engine.model.parameters()).device
    activities = [ProfilerActivity.CPU]
    sort_by = "self_cpu_time_total"
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        sort_by = "self_device_time_total"
    params = SamplingParams(temperature=0.0, max_tokens=sample.output_len, stop_token_ids=[])
    with profile(activities=activities, record_shapes=False) as prof:
        engine.generate_tokens([sample.prompt_token_ids], params)
    averages = prof.key_averages()
    table, sort_by = _profiler_table(averages, sort_by)
    events = [_event_row(event) for event in averages]
    time_key = "self_device_time_total_us" if device.type == "cuda" else "self_cpu_time_total_us"
    events.sort(key=lambda row: row.get(time_key) or 0.0, reverse=True)
    return {
        "activities": [activity.name for activity in activities],
        "sort_by": sort_by,
        "time_key": time_key,
        "prompt_len": len(sample.prompt_token_ids),
        "output_len": sample.output_len,
        "top_self_time": events[:20],
        "groups": _group_self_time(events, time_key),
        "table": table,
    }


class _TokenClock(BaseStreamer):
    """Stamp each new token. The first ``put`` from ``generate`` is the prompt."""

    def __init__(self) -> None:
        self.token_times: list[float] = []
        self._saw_prompt = False

    def put(self, value) -> None:
        now = time.perf_counter()
        if not self._saw_prompt:
            self._saw_prompt = True
            return
        # One greedy step hands over one id. A wider tensor would be several
        # tokens with one timestamp, which cannot be an inter-token gap.
        width = int(value.shape[-1]) if hasattr(value, "shape") else 1
        if width != 1:
            raise RuntimeError(f"expected one new token per step, got width {width}")
        self.token_times.append(now)

    def end(self) -> None:
        return None


def _generate_hf(model, sample: Sample, device: torch.device):
    """One ``generate`` call. Minimum and maximum new tokens are the same length."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    origin = time.perf_counter()
    clock = _TokenClock()
    input_ids = torch.tensor([sample.prompt_token_ids], dtype=torch.long, device=device)  # [1, S]
    with torch.inference_mode():
        sequences = model.generate(
            input_ids,
            max_new_tokens=sample.output_len,
            min_new_tokens=sample.output_len,
            do_sample=False,
            streamer=clock,
            pad_token_id=_pad_token_id(model),
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    n_out = int(sequences.shape[-1] - input_ids.shape[-1])
    times = clock.token_times
    if len(times) != n_out:
        raise RuntimeError(f"streamer saw {len(times)} tokens, generate returned {n_out}")
    ttft = None if not times else times[0] - origin
    e2e = (times[-1] if times else time.perf_counter()) - origin
    itl = [times[i] - times[i - 1] for i in range(1, len(times))]
    return n_out, ttft, e2e, itl


def _pad_token_id(model) -> int:
    """An id ``generate`` can write on a finished row. Generation is fixed-length."""
    eos = model.config.eos_token_id
    if isinstance(eos, (list, tuple)):
        return int(eos[0])
    if eos is None:
        return 0
    return int(eos)


def _profiler_table(averages, sort_by: str) -> tuple[str, str]:
    """The profiler's own text table. Fall back when a device sort key is absent."""
    try:
        return averages.table(sort_by=sort_by, row_limit=20), sort_by
    except (KeyError, ValueError, AttributeError):
        fallback = "self_cpu_time_total"
        return averages.table(sort_by=fallback, row_limit=20), fallback


def _event_row(event) -> dict:
    row = {"name": event.key, "count": int(event.count)}
    for name in (
        "self_cpu_time_total",
        "cpu_time_total",
        "self_device_time_total",
        "device_time_total",
    ):
        if hasattr(event, name):
            row[name + "_us"] = float(getattr(event, name))
    return row


def _group_self_time(events: list[dict], time_key: str) -> dict:
    """Coarse buckets over aten ops only.

    CUDA kernel rows repeat time that the launching aten op already counts as
    its own device time. Adding those rows again makes the fractions overlap.
    """
    groups = {"attention": 0.0, "gemm": 0.0, "norm": 0.0, "elementwise": 0.0, "other": 0.0}
    for event in events:
        if not event["name"].startswith("aten::"):
            continue
        groups[_bucket(event["name"])] += float(event.get(time_key) or 0.0)
    total = sum(groups.values())
    return {
        "time_key": time_key,
        "us": groups,
        "fraction": {name: (value / total if total else 0.0) for name, value in groups.items()},
    }


def _bucket(name: str) -> str:
    lowered = name.lower()
    if "scaled_dot_product" in lowered or "attention" in lowered or "sdpa" in lowered:
        return "attention"
    if "addmm" in lowered or lowered.endswith("::mm") or "bmm" in lowered or "linear" in lowered:
        return "gemm"
    if "norm" in lowered:
        return "norm"
    if any(token in lowered for token in ("copy", "add", "mul", "silu", "sigmoid", "softmax")):
        return "elementwise"
    return "other"


def _load_tinyserve(model_dir: Path, device: torch.device) -> Engine:
    from tinyserve.model.llama import LlamaForCausalLM, LlamaModelConfig
    from tinyserve.model.tokenizer import Tokenizer
    from tinyserve.model.weights import load_hf_weights

    model = LlamaForCausalLM(LlamaModelConfig.from_json(model_dir / "config.json")).eval()
    load_hf_weights(model, model_dir, device=device, dtype=torch.bfloat16)
    return Engine(model, Tokenizer.from_pretrained(str(model_dir)))


def _load_hf(model_dir: Path, device: torch.device):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to(device).eval()


def _drop(model, device: torch.device) -> None:
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _write_profile(path: Path, config: dict, profile_body: dict) -> None:
    env = collect_env_info()
    payload = {
        "record": "profile",
        "config": config,
        "env_info": env,
        "git_commit": env.get("git_commit"),
        "profile": profile_body,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    """Run both baselines and one profiler pass. Refuses to start without weights."""
    parser = argparse.ArgumentParser(
        description="Dev-model offline baselines and one profiler pass."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results/phase2"))
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not (args.model / "model.safetensors").exists():
        raise SystemExit(f"no weights at {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = synthetic_workload(
        num_requests=args.num_requests,
        input_len=args.input_len,
        output_len=args.output_len,
        seed=args.seed,
        vocab_size=_vocab_size(args.model),
    )
    day = datetime.now(UTC).date().isoformat()
    shared = {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "model_dir": str(args.model),
        "dtype": "bfloat16",
        "workload": "synthetic",
        "num_requests": args.num_requests,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_warmup_requests": args.warmup,
        "num_repeats": args.repeats,
        "seed": args.seed,
        "ignore_eos": True,
        "request_rate": float("inf"),
    }
    out = args.output_dir
    tiny_path = out / f"{day}_p2-3-tinyserve-offline.jsonl"
    hf_path = out / f"{day}_p2-3-hf-generate.jsonl"
    profile_path = out / f"{day}_p2-3-profiler.json"

    print("tinyserve", flush=True)
    engine = _load_tinyserve(args.model, device)
    run_offline(
        engine,
        samples,
        tiny_path,
        config={**shared, "engine": "tinyserve"},
        num_warmup_requests=args.warmup,
        num_repeats=args.repeats,
        ignore_eos=True,
    )
    print("profile", flush=True)
    _write_profile(
        profile_path, {**shared, "engine": "tinyserve"}, profile_generation(engine, samples[0])
    )
    _drop(engine, device)

    print("hf_generate", flush=True)
    hf = _load_hf(args.model, device)
    run_hf_generate(
        hf,
        samples,
        hf_path,
        config={**shared, "engine": "hf_generate"},
        num_warmup_requests=args.warmup,
        num_repeats=args.repeats,
    )
    _drop(hf, device)
    print(tiny_path)
    print(hf_path)
    print(profile_path)


def _vocab_size(model_dir: Path) -> int:
    config = json.loads((model_dir / "config.json").read_text())
    return int(config["vocab_size"])


if __name__ == "__main__":
    main()
