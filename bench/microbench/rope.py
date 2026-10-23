"""Time in-place RoPE against the PyTorch reference.

The timed Triton call writes Q and K in place. The reference returns new
tensors. Bandwidth uses algorithm bytes (each of Q and K read once and written
once) over the median ``do_bench`` time, divided by the same 256 GB/s peak as
the RMSNorm file. Nsight Compute counters are still blocked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.env_info import collect_env_info
from tinyserve.kernels.reference import apply_rope as apply_rope_ref
from tinyserve.kernels.rope import algorithm_bytes, apply_rope_
from tinyserve.model.rope import compute_inv_freq

PEAK_BANDWIDTH_BYTES_PER_S = 256_000_000_000
PEAK_SOURCE = (
    "NVIDIA GeForce RTX 4060 Laptop GPU: 8 GB GDDR6, 128-bit bus "
    "(https://www.nvidia.com/en-us/geforce/laptops/compare/). "
    "16 Gbps times 128/8 is 256 GB/s = 256e9 bytes/s."
)
# tokens, query heads, kv heads, head_dim. 32/8/64 is Llama-3.2-1B. 32/8/128 is 8B-shaped.
SHAPES = (
    (1, 32, 8, 64),
    (8, 32, 8, 64),
    (32, 32, 8, 64),
    (512, 32, 8, 64),
    (2048, 32, 8, 64),
    (1, 32, 8, 128),
    (512, 32, 8, 128),
)


def _median_ms(fn) -> float:
    """Median milliseconds from triton's bench helper."""
    from triton.testing import do_bench

    measured = do_bench(fn, warmup=25, rep=100, quantiles=(0.5,))
    if isinstance(measured, (int, float)):
        return float(measured)
    return float(measured[0])


def measure_shape(
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    repeats: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict:
    """Three timed passes. The recorded time is the median of those medians."""
    torch.manual_seed(0)
    inv_freq = compute_inv_freq(head_dim, 500000.0).to(device)
    positions = torch.arange(num_tokens, device=device, dtype=torch.int64)
    query = torch.randn(num_tokens, num_heads, head_dim, device=device, dtype=dtype)
    key = torch.randn(num_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
    apply_rope_(query, key, positions, inv_freq)
    torch.cuda.synchronize()
    fused = [
        _median_ms(lambda: apply_rope_(query, key, positions, inv_freq)) for _ in range(repeats)
    ]
    pytorch = [
        _median_ms(
            lambda: (
                apply_rope_ref(query, positions, inv_freq),
                apply_rope_ref(key, positions, inv_freq),
            )
        )
        for _ in range(repeats)
    ]
    fused_median = sorted(fused)[len(fused) // 2]
    pytorch_median = sorted(pytorch)[len(pytorch) // 2]
    moved = algorithm_bytes(num_tokens, num_heads, num_kv_heads, head_dim, dtype)
    achieved = moved / (fused_median / 1e3)
    return {
        "record": "shape",
        "num_tokens": num_tokens,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": str(dtype).removeprefix("torch."),
        "repeats": repeats,
        "fused_ms": fused,
        "pytorch_ms": pytorch,
        "fused_ms_median": fused_median,
        "pytorch_ms_median": pytorch_median,
        "algorithm_bytes": moved,
        "achieved_bandwidth_bytes_per_s": achieved,
        "peak_bandwidth_bytes_per_s": PEAK_BANDWIDTH_BYTES_PER_S,
        "bandwidth_utilization": achieved / PEAK_BANDWIDTH_BYTES_PER_S,
        "peak_source": PEAK_SOURCE,
    }


def main(argv: list[str] | None = None) -> None:
    """Write one JSONL of in-place RoPE versus the PyTorch reference."""
    parser = argparse.ArgumentParser(description="Microbenchmark in-place RoPE.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.repeats < 1:
        raise SystemExit("repeats must be at least 1")
    device = torch.device("cuda")
    rows = [
        measure_shape(*shape, repeats=args.repeats, dtype=torch.bfloat16, device=device)
        for shape in SHAPES
    ]
    env = collect_env_info()
    meta = {
        "record": "meta",
        "config": {
            "kernel": "inplace_rope",
            "dtype": "bfloat16",
            "shapes": [list(shape) for shape in SHAPES],
            "shape_fields": ["num_tokens", "num_heads", "num_kv_heads", "head_dim"],
            "repeats": args.repeats,
            "warmup": 25,
            "rep": 100,
            "bytes": "Q and K read once and written once, plus positions and inv_freq",
            "peak_bandwidth_bytes_per_s": PEAK_BANDWIDTH_BYTES_PER_S,
            "peak_source": PEAK_SOURCE,
            "nsight_compute": "not used; GPU performance counters are blocked on the Windows host",
        },
        "env_info": env,
        "git_commit": env.get("git_commit"),
    }
    ranked = sorted(rows, key=lambda row: row["bandwidth_utilization"])
    median = ranked[len(ranked) // 2]
    summary = {
        "record": "summary",
        "num_shapes": len(rows),
        "median_shape_by_utilization": [
            median["num_tokens"],
            median["num_heads"],
            median["num_kv_heads"],
            median["head_dim"],
        ],
        "median_bandwidth_utilization": median["bandwidth_utilization"],
        "median_achieved_bandwidth_bytes_per_s": median["achieved_bandwidth_bytes_per_s"],
        "highest_bandwidth_utilization": ranked[-1]["bandwidth_utilization"],
        "highest_shape": [
            ranked[-1]["num_tokens"],
            ranked[-1]["num_heads"],
            ranked[-1]["num_kv_heads"],
            ranked[-1]["head_dim"],
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [meta, *rows, summary]
    args.output.write_text("".join(json.dumps(line) + "\n" for line in lines))
    print(args.output)


if __name__ == "__main__":
    main()
