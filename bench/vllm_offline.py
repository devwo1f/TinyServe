"""Offline vLLM run on the same synthetic requests as the other baselines.

Execute this with the separate vLLM virtualenv documented in
``bench/vllm_baseline.md``. TinyServe's own environment does not install vLLM.
The import of vLLM stays inside ``main`` so a unit test can check the row
shape without that package.

vLLM's request metrics expose the first token and the last token, not each
gap between them. Inter-token latencies are therefore an empty list. TPOT is
still ``(e2e - ttft) / (num_output_tokens - 1)``, which is the mean gap.
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

import torch

from bench.datasets import Sample, synthetic_workload
from bench.offline import (
    _wall_end,
    _wall_start,
    repeat_record,
    request_record,
    write_benchmark_jsonl,
)


def latency_seconds(metrics) -> tuple[float, float]:
    """TTFT is vLLM's wall-clock first-token latency.

    ``first_token_ts`` and ``last_token_ts`` use the engine-core monotonic
    clock, which is not ``arrival_time``. The gap between those two stamps is
    added to the wall-clock TTFT so end-to-end latency stays on one timeline.
    """
    ttft = getattr(metrics, "first_token_latency", None)
    first = getattr(metrics, "first_token_ts", None)
    last = getattr(metrics, "last_token_ts", None)
    if ttft is None or not first or not last:
        raise RuntimeError(
            "vLLM request metrics are missing first-token latency or token timestamps"
        )
    return float(ttft), float(ttft) + float(last - first)


def rows_for_repeat(
    samples: list[Sample], repeat: int, measured: list[tuple[int, float, float]]
) -> list[dict]:
    """One repeat of ``(num_output_tokens, ttft_s, e2e_s)`` into Section 11 request rows."""
    if len(measured) != len(samples):
        raise ValueError(f"expected {len(samples)} results, got {len(measured)}")
    rows = []
    for index, (sample, (n_out, ttft_s, e2e_s)) in enumerate(zip(samples, measured, strict=True)):
        if n_out != sample.output_len:
            raise RuntimeError(
                f"request {index} emitted {n_out} tokens, dataset length is {sample.output_len}"
            )
        rows.append(
            request_record(
                repeat,
                index,
                sample,
                num_output_tokens=n_out,
                ttft_s=ttft_s,
                e2e_s=e2e_s,
                itl_s=[],
            )
        )
    return rows


def _prompts(samples: list[Sample]) -> list[dict]:
    """Token ids only. Re-encoding the empty synthetic prompt would not be these ids."""
    return [{"prompt_token_ids": list(sample.prompt_token_ids)} for sample in samples]


def _generate(llm, samples: list[Sample], sampling):
    return llm.generate(_prompts(samples), sampling, use_tqdm=False)


def _measured(outputs, samples: list[Sample]) -> list[tuple[int, float, float]]:
    measured = []
    for sample, output in zip(samples, outputs, strict=True):
        token_ids = list(output.outputs[0].token_ids)
        ttft_s, e2e_s = latency_seconds(output.metrics)
        measured.append((len(token_ids), ttft_s, e2e_s))
        if len(token_ids) != sample.output_len:
            raise RuntimeError(
                f"vLLM emitted {len(token_ids)} tokens, dataset length is {sample.output_len}"
            )
    return measured


def run_vllm(
    llm,
    samples: list[Sample],
    sampling,
    output_path: str | Path,
    *,
    config: dict,
    num_warmup_requests: int,
    num_repeats: int,
) -> dict:
    """Warm up, then time each batched ``generate``. One call holds every sample."""
    if not samples:
        raise ValueError("samples is empty")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warmup = samples[:num_warmup_requests]
    if warmup:
        _generate(llm, warmup, sampling)
    request_rows: list[dict] = []
    repeat_rows: list[dict] = []
    for repeat in range(num_repeats):
        wall_start = _wall_start(device)
        outputs = _generate(llm, samples, sampling)
        wall_s = _wall_end(device, wall_start)
        measured_rows = rows_for_repeat(samples, repeat, _measured(outputs, samples))
        repeat_rows.append(repeat_record(repeat, measured_rows, wall_s))
        request_rows.extend(measured_rows)
    return write_benchmark_jsonl(
        output_path,
        config=config,
        request_rows=request_rows,
        repeat_rows=repeat_rows,
        num_warmup_requests=num_warmup_requests,
    )


def main(argv: list[str] | None = None) -> None:
    """Load vLLM on the local checkpoint and write one Section 11 file."""
    parser = argparse.ArgumentParser(description="vLLM offline baseline on the dev model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results/phase2"))
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    args = parser.parse_args(argv)
    if not (args.model / "config.json").exists():
        raise SystemExit(f"no model config at {args.model}")

    # FlashInfer's sampler JIT-compiles with nvcc. This machine has no CUDA toolkit.
    # Greedy sampling does not need that kernel. Set this before vLLM is imported.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import vllm
    from vllm import LLM, SamplingParams

    samples = synthetic_workload(
        num_requests=args.num_requests,
        input_len=args.input_len,
        output_len=args.output_len,
        seed=args.seed,
        vocab_size=_vocab_size(args.model),
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_len,
        ignore_eos=True,
    )
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=False,
        trust_remote_code=False,
        # The offline LLM class turns stats off, which leaves request metrics empty.
        disable_log_stats=False,
    )
    day = datetime.now(UTC).date().isoformat()
    path = args.output_dir / f"{day}_p2-4-vllm-offline.jsonl"
    config = {
        "engine": "vllm",
        "vllm_version": vllm.__version__,
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
        "temperature": 0.0,
        "request_rate": float("inf"),
        "batched": True,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") == "1",
    }
    run_vllm(
        llm,
        samples,
        sampling,
        path,
        config=config,
        num_warmup_requests=args.warmup,
        num_repeats=args.repeats,
    )
    print(path)


def _vocab_size(model_dir: Path) -> int:
    import json

    return int(json.loads((model_dir / "config.json").read_text())["vocab_size"])


if __name__ == "__main__":
    main()
