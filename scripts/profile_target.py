"""One greedy request for the Nsight wrappers.

Nsight Systems and Nsight Compute need a command that is the naive engine and
nothing else. Weight loading stays outside the CUDA profiler range so the
timeline is the generation, not the copy of the checkpoint onto the GPU.
A warmup of the same shape runs first, so the captured request is not the
first time those kernel shapes are launched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bench.datasets import synthetic_workload
from tinyserve.engine.sequence import SamplingParams


def main(argv: list[str] | None = None) -> None:
    """Load the dev model and generate one fixed-length request inside the profiler range."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("models/Llama-3.2-1B-Instruct"))
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target-json",
        type=Path,
        default=Path("profiles/last_target.json"),
        help="Sidecar describing the request the profiler just captured.",
    )
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("profile_target needs a CUDA GPU")
    if not (args.model / "model.safetensors").is_file():
        raise SystemExit(f"no weights at {args.model}")

    engine = _load(args.model)
    sample = synthetic_workload(
        num_requests=1,
        input_len=args.input_len,
        output_len=args.output_len,
        seed=args.seed,
        vocab_size=_vocab_size(args.model),
    )[0]
    params = SamplingParams(temperature=0.0, max_tokens=args.output_len, seed=args.seed)
    engine.generate_tokens([sample.prompt_token_ids], params)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    try:
        result = engine.generate_tokens([sample.prompt_token_ids], params)
        torch.cuda.synchronize()
    finally:
        torch.cuda.profiler.stop()

    args.target_json.parent.mkdir(parents=True, exist_ok=True)
    args.target_json.write_text(
        json.dumps(
            {
                "model": str(args.model),
                "input_len": args.input_len,
                "output_len": args.output_len,
                "seed": args.seed,
                "dtype": "bfloat16",
                "temperature": 0.0,
                "num_output_tokens": len(result[0].output_token_ids),
                "warmup": "one uncaptured request of the same shape",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"output_tokens {len(result[0].output_token_ids)}")


def _vocab_size(model_dir: Path) -> int:
    config = json.loads((model_dir / "config.json").read_text())
    return int(config["vocab_size"])


def _load(model_dir: Path):
    from tinyserve.engine.engine import Engine
    from tinyserve.model.llama import LlamaForCausalLM, LlamaModelConfig
    from tinyserve.model.tokenizer import Tokenizer
    from tinyserve.model.weights import load_hf_weights

    device = torch.device("cuda")
    model = LlamaForCausalLM(LlamaModelConfig.from_json(model_dir / "config.json")).eval()
    load_hf_weights(model, model_dir, device=device, dtype=torch.bfloat16)
    return Engine(model, Tokenizer.from_pretrained(str(model_dir)))


if __name__ == "__main__":
    main()
