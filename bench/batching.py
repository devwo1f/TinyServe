"""Continuous-batching measurements against the serial engine.

``run_offline`` starts the next request when the previous one finishes. That
is the naive engine. ``run_concurrent`` puts the requests of one repeat into
a single step loop, which is the throughput the scheduler can reach.

Poisson offsets are seconds from the start of the measured pass. The first
request arrives at 0. Later requests wait, and ``step`` keeps running during
that wait so a request already in flight is not paused.
"""

from __future__ import annotations

import argparse
import gc
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from bench.datasets import Sample, synthetic_workload
from bench.offline import (
    _wall_end,
    _wall_start,
    repeat_record,
    request_record,
    run_offline,
    write_benchmark_jsonl,
)
from tinyserve.config import SchedulerConfig
from tinyserve.engine.engine import Engine
from tinyserve.engine.sequence import SamplingParams, Sequence, SequenceStatus


def poisson_offsets(num_requests: int, request_rate: float, seed: int) -> list[float]:
    """Arrival times for a Poisson process, in seconds. The first is 0.

    The gaps are exponential with mean ``1 / request_rate``. One seed rebuilds
    the same schedule, so repeats differ by the engine, not by a new draw.
    """
    if num_requests < 1:
        raise ValueError("num_requests must be at least 1")
    if request_rate <= 0:
        raise ValueError("request_rate must be positive")
    rng = np.random.default_rng(seed)
    offsets = [0.0]
    for _ in range(num_requests - 1):
        offsets.append(offsets[-1] + float(rng.exponential(1.0 / request_rate)))
    return offsets


def run_concurrent(
    engine: Engine,
    samples: list[Sample],
    output_path: str | Path,
    *,
    config: dict,
    num_warmup_requests: int = 0,
    num_repeats: int = 1,
    arrival_offsets: list[float] | None = None,
) -> dict:
    """Time one repeat as a single step loop and write the Section 11 file.

    ``arrival_offsets is None`` admits every request before the first step.
    A list admits each request when its offset has elapsed.
    """
    if not samples:
        raise ValueError("samples is empty")
    if num_warmup_requests < 0:
        raise ValueError("num_warmup_requests must be >= 0")
    if num_repeats < 1:
        raise ValueError("num_repeats must be at least 1")
    params = _params_for(samples)
    prompts = [list(sample.prompt_token_ids) for sample in samples]
    # Allocate before the clock. The first measured step should not pay for the pool.
    engine.prepare_pool(prompts, params)
    warmup = samples[:num_warmup_requests]
    device = next(engine.model.parameters()).device
    request_rows: list[dict] = []
    repeat_rows: list[dict] = []
    for repeat in range(num_repeats):
        if warmup:
            engine.generate_tokens(
                [list(sample.prompt_token_ids) for sample in warmup], _params_for(warmup)
            )
        print(f"{config.get('label', 'run')} repeat {repeat + 1}/{num_repeats}", flush=True)
        wall_start = _wall_start(device)
        if arrival_offsets is None:
            results = engine.generate_tokens(prompts, params)
        else:
            results = run_arrivals(engine, prompts, params, arrival_offsets)
        wall_s = _wall_end(device, wall_start)
        measured = _rows(samples, repeat, results)
        repeat_rows.append(repeat_record(repeat, measured, wall_s))
        request_rows.extend(measured)
    return write_benchmark_jsonl(
        output_path,
        config=config,
        request_rows=request_rows,
        repeat_rows=repeat_rows,
        num_warmup_requests=num_warmup_requests,
    )


def run_arrivals(
    engine: Engine,
    prompts: list[list[int]],
    params: SamplingParams,
    offsets: list[float],
) -> list:
    """Submit each prompt at its offset and step until all of them finish."""
    if len(offsets) != len(prompts):
        raise ValueError("arrival offsets must have one time per prompt")
    if any(offset < 0 for offset in offsets):
        raise ValueError("arrival offsets must be >= 0")
    engine.prepare_pool(prompts, params)
    order = sorted(range(len(prompts)), key=lambda index: (offsets[index], index))
    sequences: list[Sequence | None] = [None] * len(prompts)
    start = time_seconds()
    next_k = 0
    steps = 0
    cap = sum(len(prompt) + params.max_tokens for prompt in prompts) * len(prompts) * 8
    while any(seq is None or seq.status != SequenceStatus.FINISHED for seq in sequences):
        now = time_seconds() - start
        while next_k < len(order) and offsets[order[next_k]] <= now:
            index = order[next_k]
            sequences[index] = engine.add_request(prompts[index], params)
            next_k += 1
        if engine.has_pending():
            batch = engine.step()
            steps += 1
            if steps > max(cap, 1):
                raise RuntimeError("engine step did not finish the batch")
            if not batch.seqs and not batch.preempted:
                raise RuntimeError("no token was scheduled; pool or token budget is too small")
            continue
        if next_k >= len(order):
            break
        delay = offsets[order[next_k]] - (time_seconds() - start)
        if delay > 0:
            sleep(delay)
    done: list[Sequence] = []
    for seq in sequences:
        if seq is None:
            raise RuntimeError("an arrival was never submitted")
        done.append(seq)
    return engine.results(done)


def time_seconds() -> float:
    """Clock for arrivals. Split out so a test can replace it."""
    return time.perf_counter()


def sleep(seconds: float) -> None:
    """Wait until the next arrival. Split out so a test can skip the pause."""
    time.sleep(seconds)


def _params_for(samples: list[Sample]) -> SamplingParams:
    """Greedy, fixed length, no stop ids. That is this bench's ignore-EOS."""
    length = samples[0].output_len
    if any(sample.output_len != length for sample in samples):
        raise ValueError("every sample in one call needs the same output_len")
    return SamplingParams(temperature=0.0, max_tokens=length, stop_token_ids=[])


def _rows(samples: list[Sample], repeat: int, results: list) -> list[dict]:
    if len(results) != len(samples):
        raise ValueError(f"expected {len(samples)} results, got {len(results)}")
    rows = []
    for index, (sample, result) in enumerate(zip(samples, results, strict=True)):
        n_out = len(result.output_token_ids)
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
                ttft_s=result.ttft_s,
                e2e_s=result.e2e_s,
                itl_s=list(result.itl_s or []),
            )
        )
    return rows


def _engine(
    model,
    tokenizer,
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int,
) -> Engine:
    """Prefix caching off, so a later repeat does not reuse the previous prefill."""
    return Engine(
        model,
        tokenizer,
        enable_prefix_caching=False,
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=True,
        ),
    )


def _release(device: torch.device) -> None:
    """Collect after the caller has dropped its Engine reference."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def greedy_ids_match(model, tokenizer, samples: list[Sample]) -> bool:
    """True when one shared step emits the same greedy ids as one-at-a-time.

    This is not part of the timed pass. bf16 can diverge with batch shape;
    the result config records the bool instead of hiding a mismatch.
    """
    params = _params_for(samples)
    prompts = [list(sample.prompt_token_ids) for sample in samples]
    serial = _engine(model, tokenizer, max_num_seqs=1, max_num_batched_tokens=2048)
    one = [serial.generate_tokens([prompt], params)[0].output_token_ids for prompt in prompts]
    device = next(model.parameters()).device
    del serial
    _release(device)
    batched = _engine(
        model, tokenizer, max_num_seqs=max(len(prompts), 1), max_num_batched_tokens=2048
    )
    many = [row.output_token_ids for row in batched.generate_tokens(prompts, params)]
    del batched
    _release(device)
    return one == many


def _load(model_dir: Path, device: torch.device):
    from tinyserve.model.llama import LlamaForCausalLM, LlamaModelConfig
    from tinyserve.model.tokenizer import Tokenizer
    from tinyserve.model.weights import load_hf_weights

    model = LlamaForCausalLM(LlamaModelConfig.from_json(model_dir / "config.json")).eval()
    load_hf_weights(model, model_dir, device=device, dtype=torch.bfloat16)
    return model, Tokenizer.from_pretrained(str(model_dir))


def _vocab_size(model_dir: Path) -> int:
    import json

    return int(json.loads((model_dir / "config.json").read_text())["vocab_size"])


def _base_config(args, samples_note: dict) -> dict:
    return {
        "engine": "tinyserve",
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "model_dir": str(args.model),
        "dtype": "bfloat16",
        "workload": "synthetic",
        "num_warmup_requests": args.warmup,
        "num_repeats": args.repeats,
        "seed": args.seed,
        "ignore_eos": True,
        "temperature": 0.0,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        **samples_note,
    }


def main(argv: list[str] | None = None) -> None:
    """Write the Phase 4 TinyServe result files. vLLM is a separate process."""
    parser = argparse.ArgumentParser(description="Continuous-batching sweeps on the dev model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results/phase4"))
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--chunk-input-len", type=int, default=128)
    parser.add_argument("--chunk-output-len", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rates", default="1,4,16")
    parser.add_argument("--chunk-budgets", default="32,128,512")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("P4.4 measurements need the GPU")
    if not (args.model / "config.json").exists():
        raise SystemExit(f"no model config at {args.model}")

    device = torch.device("cuda")
    vocab = _vocab_size(args.model)
    samples = synthetic_workload(
        num_requests=args.num_requests,
        input_len=args.input_len,
        output_len=args.output_len,
        seed=args.seed,
        vocab_size=vocab,
    )
    chunk_samples = synthetic_workload(
        num_requests=args.num_requests,
        input_len=args.chunk_input_len,
        output_len=args.chunk_output_len,
        seed=args.seed + 1,
        vocab_size=vocab,
    )
    model, tokenizer = _load(args.model, device)
    matched = greedy_ids_match(model, tokenizer, samples)
    print(f"greedy_ids_match_serial {matched}", flush=True)
    day = datetime.now(UTC).date().isoformat()
    out = args.output_dir
    shape = {
        "num_requests": args.num_requests,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "greedy_ids_match_serial": matched,
    }
    naive = _engine(model, tokenizer, max_num_seqs=1, max_num_batched_tokens=2048)
    naive_config = _base_config(args, shape)
    naive_config.update(
        {
            "series": "offline",
            "label": "naive",
            "batched": False,
            "request_rate": float("inf"),
            "max_num_seqs": 1,
            "max_num_batched_tokens": 2048,
        }
    )
    path = out / f"{day}_p4-4-naive-offline.jsonl"
    run_offline(
        naive,
        samples,
        path,
        config=naive_config,
        num_warmup_requests=args.warmup,
        num_repeats=args.repeats,
        ignore_eos=True,
    )
    print(path, flush=True)
    del naive
    _release(device)

    batched_config = _base_config(args, shape)
    batched_config.update(
        {
            "series": "offline",
            "label": "batched",
            "batched": True,
            "request_rate": float("inf"),
            "max_num_seqs": args.num_requests,
            "max_num_batched_tokens": 2048,
        }
    )
    _run_labeled(
        model,
        tokenizer,
        samples,
        out / f"{day}_p4-4-batched-offline.jsonl",
        batched_config,
        args,
        max_num_seqs=args.num_requests,
        max_num_batched_tokens=2048,
        arrival_offsets=None,
        device=device,
    )
    for rate in _floats(args.rates):
        offsets = poisson_offsets(len(samples), rate, args.seed)
        config = _base_config(args, shape)
        config.update(
            {
                "series": "poisson",
                "label": str(rate),
                "batched": True,
                "request_rate": rate,
                "max_num_seqs": args.num_requests,
                "max_num_batched_tokens": 2048,
            }
        )
        _run_labeled(
            model,
            tokenizer,
            samples,
            out / f"{day}_p4-4-poisson-r{rate:g}.jsonl",
            config,
            args,
            max_num_seqs=args.num_requests,
            max_num_batched_tokens=2048,
            arrival_offsets=offsets,
            device=device,
        )
    chunk_shape = {
        "num_requests": args.num_requests,
        "input_len": args.chunk_input_len,
        "output_len": args.chunk_output_len,
        "seed": args.seed + 1,
    }
    for budget in _ints(args.chunk_budgets):
        config = _base_config(args, chunk_shape)
        config.update(
            {
                "series": "chunk",
                "label": str(budget),
                "batched": True,
                "request_rate": float("inf"),
                "max_num_seqs": args.num_requests,
                "max_num_batched_tokens": budget,
            }
        )
        _run_labeled(
            model,
            tokenizer,
            chunk_samples,
            out / f"{day}_p4-4-chunk-b{budget}.jsonl",
            config,
            args,
            max_num_seqs=args.num_requests,
            max_num_batched_tokens=budget,
            arrival_offsets=None,
            device=device,
        )


def _run_labeled(
    model,
    tokenizer,
    samples: list[Sample],
    path: Path,
    config: dict,
    args,
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    arrival_offsets: list[float] | None,
    device: torch.device,
) -> None:
    engine = _engine(
        model,
        tokenizer,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    run_concurrent(
        engine,
        samples,
        path,
        config=config,
        num_warmup_requests=args.warmup,
        num_repeats=args.repeats,
        arrival_offsets=arrival_offsets,
    )
    print(path, flush=True)
    del engine
    _release(device)


def _floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part]


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


if __name__ == "__main__":
    main()
