"""TTFT with and without the prefix cache on a shared-prefix workload.

One process, one copy of the model. The uncached engine runs first and is
dropped before the cached engine allocates its pool, so the two KV pools are
not live together. The script writes the Section 11 JSONL. It refuses to
write when the cached run does not compute fewer prompt tokens, or when the
greedy tokens differ: that would be a wrong cache, not a speedup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bench.datasets import shared_prefix_workload
from bench.offline import _json_ready, _percentiles
from scripts.env_info import collect_env_info
from tinyserve.engine.engine import Engine
from tinyserve.engine.sequence import SamplingParams
from tinyserve.model.llama import LlamaForCausalLM, LlamaModelConfig
from tinyserve.model.tokenizer import Tokenizer
from tinyserve.model.weights import load_hf_weights

PREFIX = "You are a careful assistant. Answer in one short sentence. " * 8
SUFFIXES = ["Say north.", "Say south.", "Say east.", "Say west."]


def _rows(engine: Engine, samples, output_len: int) -> list[dict]:
    params = SamplingParams(temperature=0.0, max_tokens=output_len)
    prompts = [sample.prompt_token_ids for sample in samples]
    results = engine.generate_tokens(prompts, params)
    rows = []
    for index, (sample, result) in enumerate(zip(samples, results, strict=True)):
        prompt_len = len(sample.prompt_token_ids)
        rows.append(
            {
                "record": "request",
                "index": index,
                "workload": "shared_prefix",
                "prefix_caching": engine.enable_prefix_caching,
                "prompt_len": prompt_len,
                "cached_prompt_tokens": result.num_cached_prompt_tokens,
                "computed_prompt_tokens": prompt_len - result.num_cached_prompt_tokens,
                "num_output_tokens": len(result.output_token_ids),
                "output_token_ids": result.output_token_ids,
                "ttft_s": result.ttft_s,
                "e2e_s": result.e2e_s,
            }
        )
    return rows


def _check(off: list[dict], on: list[dict]) -> None:
    if sum(row["computed_prompt_tokens"] for row in on) >= sum(
        row["computed_prompt_tokens"] for row in off
    ):
        raise SystemExit("prefix caching did not reduce computed prompt tokens")
    for left, right in zip(off, on, strict=True):
        if left["output_token_ids"] != right["output_token_ids"]:
            raise SystemExit(f"request {left['index']} tokens differ with the prefix cache")


def _summary(off: list[dict], on: list[dict]) -> dict:
    def side(rows: list[dict]) -> dict:
        return {
            "computed_prompt_tokens": sum(row["computed_prompt_tokens"] for row in rows),
            "cached_prompt_tokens": sum(row["cached_prompt_tokens"] for row in rows),
            "ttft_s": _percentiles([row["ttft_s"] for row in rows if row["ttft_s"] is not None]),
        }

    return {"record": "summary", "prefix_caching_off": side(off), "prefix_caching_on": side(on)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/Llama-3.2-1B-Instruct")
    parser.add_argument("--output-len", type=int, default=8)
    parser.add_argument(
        "--output",
        default="docs/results/phase3/2026-10-07_p3-6-prefix-cache.jsonl",
    )
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    if not (model_dir / "model.safetensors").exists():
        raise SystemExit(f"missing weights in {model_dir}")

    tokenizer = Tokenizer.from_pretrained(str(model_dir))
    samples = shared_prefix_workload(
        PREFIX,
        SUFFIXES,
        tokenizer.encode,
        output_len=args.output_len,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = LlamaForCausalLM(LlamaModelConfig.from_json(model_dir / "config.json")).eval()
    load_hf_weights(model, model_dir, device=device, dtype=dtype)

    off_engine = Engine(model, tokenizer, enable_prefix_caching=False)
    off = _rows(off_engine, samples, args.output_len)
    del off_engine
    if device.type == "cuda":
        torch.cuda.empty_cache()
    on_engine = Engine(model, tokenizer, enable_prefix_caching=True)
    on = _rows(on_engine, samples, args.output_len)
    _check(off, on)

    env = collect_env_info()
    meta = {
        "record": "meta",
        "config": {
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "model_dir": str(model_dir),
            "dtype": "bfloat16" if dtype == torch.bfloat16 else "float32",
            "workload": "shared_prefix",
            "num_requests": len(samples),
            "output_len": args.output_len,
            "block_size": on_engine.block_size,
            "prefix_tokens": len(tokenizer.encode(PREFIX)),
            "seed": 0,
            "temperature": 0,
        },
        "env_info": env,
        "git_commit": env.get("git_commit"),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [meta, *off, *on, _summary(off, on)]
    path.write_text("".join(json.dumps(_json_ready(line)) + "\n" for line in lines))
    print(path)


if __name__ == "__main__":
    main()
