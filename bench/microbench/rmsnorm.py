"""Time fused add+RMSNorm against the PyTorch reference and record bandwidth.

Achieved bandwidth is algorithm bytes divided by the median kernel time.
Algorithm bytes count each tensor once. The kernel also reloads the row, so
this is not an Nsight Compute DRAM counter. Those counters are still blocked
on the Windows host.

Peak bandwidth is 256 GB/s for this GPU. NVIDIA's laptop compare page lists
the GeForce RTX 4060 Laptop GPU as 8 GB GDDR6 on a 128-bit bus
(https://www.nvidia.com/en-us/geforce/laptops/compare/). That part's memory
rate is 16 Gbps, and 128/8 * 16 = 256 GB/s. The page does not print the GB/s
figure; 256e9 is that product, in decimal bytes per second.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.env_info import collect_env_info
from tinyserve.kernels.reference import add_rmsnorm as add_rmsnorm_ref
from tinyserve.kernels.rmsnorm import add_rmsnorm, algorithm_bytes

# Decimal GB/s, matching how memory bandwidth is quoted on spec sheets.
PEAK_BANDWIDTH_BYTES_PER_S = 256_000_000_000
PEAK_SOURCE = (
    "NVIDIA GeForce RTX 4060 Laptop GPU: 8 GB GDDR6, 128-bit bus "
    "(https://www.nvidia.com/en-us/geforce/laptops/compare/). "
    "16 Gbps times 128/8 is 256 GB/s = 256e9 bytes/s."
)
SHAPES = (
    (1, 2048),
    (8, 2048),
    (32, 2048),
    (512, 2048),
    (2048, 2048),
    (1, 4096),
    (32, 4096),
)


def _median_ms(fn) -> float:
    """Median milliseconds from triton's bench helper.

    ``quantiles`` makes the helper return a list. One number is the median.
    """
    from triton.testing import do_bench

    measured = do_bench(fn, warmup=25, rep=100, quantiles=(0.5,))
    if isinstance(measured, (int, float)):
        return float(measured)
    return float(measured[0])


def measure_shape(
    rows: int, hidden: int, *, repeats: int, dtype: torch.dtype, device: torch.device
) -> dict:
    """Three timed passes. The recorded time is the median of those medians."""
    torch.manual_seed(0)
    x = torch.randn(rows, hidden, device=device, dtype=dtype)
    residual = torch.randn(rows, hidden, device=device, dtype=dtype)
    weight = torch.randn(hidden, device=device, dtype=dtype)
    eps = 1e-5
    # Compile before the timed calls so the first repeat is not a cold launch.
    add_rmsnorm(x, residual, weight, eps)
    torch.cuda.synchronize()
    fused = [_median_ms(lambda: add_rmsnorm(x, residual, weight, eps)) for _ in range(repeats)]
    pytorch = [
        _median_ms(lambda: add_rmsnorm_ref(x, residual, weight, eps)) for _ in range(repeats)
    ]
    fused_median = sorted(fused)[len(fused) // 2]
    pytorch_median = sorted(pytorch)[len(pytorch) // 2]
    moved = algorithm_bytes(rows, hidden, dtype)
    seconds = fused_median / 1e3
    achieved = moved / seconds
    return {
        "record": "shape",
        "rows": rows,
        "hidden": hidden,
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
    """Write one JSONL of fused-versus-PyTorch times and bandwidth."""
    parser = argparse.ArgumentParser(description="Microbenchmark fused add+RMSNorm.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.repeats < 1:
        raise SystemExit("repeats must be at least 1")
    device = torch.device("cuda")
    rows = [
        measure_shape(rows, hidden, repeats=args.repeats, dtype=torch.bfloat16, device=device)
        for rows, hidden in SHAPES
    ]
    env = collect_env_info()
    meta = {
        "record": "meta",
        "config": {
            "kernel": "fused_add_rmsnorm",
            "dtype": "bfloat16",
            "shapes": [list(shape) for shape in SHAPES],
            "repeats": args.repeats,
            "warmup": 25,
            "rep": 100,
            "bytes": "algorithm_bytes: x, residual, weight once, plus the sum and the norm",
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
        "median_shape_by_utilization": [median["rows"], median["hidden"]],
        "median_bandwidth_utilization": median["bandwidth_utilization"],
        "median_achieved_bandwidth_bytes_per_s": median["achieved_bandwidth_bytes_per_s"],
        "highest_bandwidth_utilization": ranked[-1]["bandwidth_utilization"],
        "highest_shape": [ranked[-1]["rows"], ranked[-1]["hidden"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [meta, *rows, summary]
    args.output.write_text("".join(json.dumps(line) + "\n" for line in lines))
    print(args.output)


if __name__ == "__main__":
    main()
