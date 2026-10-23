"""Draw Phase 4 figures from Section 11 JSONL files.

Numbers on the axes are read from those files. This module does not measure
anything, and it does not accept hand-typed series.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_run(path: Path) -> tuple[dict, dict]:
    """Return ``(meta, summary)`` from one result file."""
    meta = None
    summary = None
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("record") == "meta":
            meta = row
        elif row.get("record") == "summary":
            summary = row
    if meta is None or summary is None:
        raise ValueError(f"{path} is missing a meta or summary line")
    return meta, summary


def plot_directory(result_dir: Path, dest_dir: Path) -> list[Path]:
    """Write the offline, Poisson, and chunk figures that have data."""
    offline: list[tuple[str, float]] = []
    poisson: list[tuple[float, dict]] = []
    chunk: list[tuple[int, dict]] = []
    for path in sorted(result_dir.glob("*.jsonl")):
        meta, summary = read_run(path)
        config = meta["config"]
        series = config.get("series")
        if series == "offline":
            offline.append((str(config["label"]), float(summary["output_throughput"])))
        elif series == "poisson":
            poisson.append((float(config["request_rate"]), summary))
        elif series == "chunk":
            chunk.append((int(config["max_num_batched_tokens"]), summary))
    written: list[Path] = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    if offline:
        order = {"naive": 0, "batched": 1, "vllm": 2}
        offline.sort(key=lambda item: order.get(item[0], 9))
        dest = dest_dir / "p4-4-offline-throughput.png"
        plot_bars(offline, dest, ylabel="output tokens / s", title="Offline output throughput")
        written.append(dest)
    if poisson:
        poisson.sort(key=lambda item: item[0])
        dest = dest_dir / "p4-4-poisson.png"
        _plot_tradeoff(
            [item[0] for item in poisson],
            [float(item[1]["output_throughput"]) for item in poisson],
            [_p99_itl(item[1]) for item in poisson],
            dest,
            xlabel="request rate (1/s)",
            title="Poisson arrivals",
        )
        written.append(dest)
    if chunk:
        chunk.sort(key=lambda item: item[0])
        dest = dest_dir / "p4-4-chunk-itl.png"
        _plot_tradeoff(
            [item[0] for item in chunk],
            [float(item[1]["output_throughput"]) for item in chunk],
            [_p99_itl(item[1]) for item in chunk],
            dest,
            xlabel="max batched tokens",
            title="Chunk size",
        )
        written.append(dest)
    return written


def plot_bars(runs: list[tuple[str, float]], dest: Path, *, ylabel: str, title: str) -> None:
    """One bar per engine. The height is that file's summary throughput."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([label for label, _ in runs], [value for _, value in runs])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)


def _plot_tradeoff(
    xs: list[float],
    throughput: list[float],
    itl_p99: list[float],
    dest: Path,
    *,
    xlabel: str,
    title: str,
) -> None:
    """Throughput and p99 inter-token latency on one chart, two scales."""
    plt = _pyplot()
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(xs, throughput, marker="o", color="C0", label="output throughput")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("output tokens / s")
    ax1.set_title(title)
    ax2 = ax1.twinx()
    ax2.plot(xs, itl_p99, marker="s", color="C1", label="p99 ITL")
    ax2.set_ylabel("p99 ITL (s)")
    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)


def _p99_itl(summary: dict) -> float:
    value = summary["itl_s"]["p99"]
    if value is None:
        raise ValueError("summary itl p99 is missing")
    return float(value)


def _pyplot():
    """Agg has no window. Importing pyplot before ``use`` would open one."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def main(argv: list[str] | None = None) -> None:
    """Read a result directory and write the png files next to it, or elsewhere."""
    parser = argparse.ArgumentParser(description="Plot Phase 4 result files.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in plot_directory(args.results, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
