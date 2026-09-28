"""Print the software and hardware environment TinyServe is running on.

Benchmark numbers are only meaningful next to the exact GPU, driver, CUDA, PyTorch, and Triton
versions that produced them (spec Sections 5 and 6), so every result file embeds the dict
returned by `collect_env_info()`. The script must work on CPU-only machines and when torch or
triton are missing, reporting `None` for unavailable fields instead of crashing.

Usage:
    uv run python scripts/env_info.py          # human-readable
    uv run python scripts/env_info.py --json   # machine-readable
"""

import argparse
import datetime
import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FIELDS = [
    "timestamp_utc",
    "git_commit",
    "git_dirty",
    "python",
    "platform",
    "cpu",
    "torch",
    "torch_cuda",
    "cudnn",
    "triton",
    "cuda_available",
    "gpu_count",
    "gpu_name",
    "gpu_memory_gib",
    "gpu_compute_capability",
    "driver",
]


def _run(cmd: list[str]) -> str | None:
    """Run a command and return stripped stdout, or None if it is missing or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=REPO_ROOT)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _module_version(name: str) -> str | None:
    """Return a module's __version__, or None if it cannot be imported."""
    try:
        return importlib.import_module(name).__version__
    except Exception:
        return None


def _git_info() -> tuple[str | None, bool | None]:
    """Commit hash and whether the working tree has uncommitted changes."""
    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return commit, (None if status is None else bool(status))


def _driver_version() -> str | None:
    """NVIDIA driver version; torch does not expose it, so ask nvidia-smi."""
    out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return out.splitlines()[0].strip() if out else None


def _torch_info() -> dict:
    """Torch, CUDA, and GPU fields. Empty dict values stay None if torch is unavailable."""
    info: dict = {}
    try:
        import torch
    except Exception:
        return info

    info["torch"] = torch.__version__
    info["torch_cuda"] = torch.version.cuda
    info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = props.name
        info["gpu_memory_gib"] = round(props.total_memory / 2**30, 2)
        info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
    else:
        info["gpu_count"] = 0
    return info


def collect_env_info() -> dict:
    """Collect every field in FIELDS; unavailable ones are None."""
    info: dict = dict.fromkeys(FIELDS)
    info["timestamp_utc"] = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    info["git_commit"], info["git_dirty"] = _git_info()
    info["python"] = sys.version.split()[0]
    info["platform"] = platform.platform()
    info["cpu"] = platform.processor() or platform.machine()
    info["triton"] = _module_version("triton")
    info["driver"] = _driver_version()
    info.update(_torch_info())
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args()

    info = collect_env_info()
    if args.json:
        print(json.dumps(info, indent=2))
        return
    width = max(len(k) for k in info)
    for key, value in info.items():
        print(f"{key:<{width}}  {value}")


if __name__ == "__main__":
    main()
