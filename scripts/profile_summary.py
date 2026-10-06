"""Turn Nsight reports into a small JSON file.

``.nsys-rep``, ``.sqlite``, and ``.ncu-rep`` files are large and gitignored.
A learning note may only quote numbers a script wrote, so this module reads
those reports and writes ``docs/results``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from scripts.env_info import collect_env_info

_KERNEL_TABLES = (
    "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
    "CUPTI_ACTIVITY_KIND_KERNEL",
)
_NAME_COLUMNS = ("demangledName", "shortName", "mangledName")
_SKIP_NCU_COLUMNS = {
    "ID",
    "Process ID",
    "Process Name",
    "Host Name",
    "Kernel Name",
    "Context",
    "Stream",
    "Block Size",
    "Grid Size",
    "Device",
    "CC",
    "Section Name",
}
_DURATION_TO_SECONDS = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def coverage_ns(intervals: list[tuple[int, int]]) -> tuple[int, int, int]:
    """Span, union of busy time, and the idle gaps inside that span.

    Overlapping kernels count once. The span runs from the first start to the
    last end, so idle is time no kernel was in flight between those stamps.
    """
    if not intervals:
        return (0, 0, 0)
    ordered = sorted(intervals)
    for start, end in ordered:
        if end < start:
            raise ValueError(f"kernel interval ends before it starts: {start} {end}")
    span_start = ordered[0][0]
    span_end = max(end for _, end in ordered)
    busy = 0
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            busy += cur_end - cur_start
            cur_start, cur_end = start, end
    busy += cur_end - cur_start
    span = span_end - span_start
    return span, busy, span - busy


def summarize_kernel_rows(rows: list[tuple[int, int, str]]) -> dict:
    """Collapse kernel ``(start_ns, end_ns, name)`` rows into one summary."""
    if not rows:
        raise ValueError("nsys report has no kernel rows")
    span_ns, busy_ns, idle_ns = coverage_ns([(start, end) for start, end, _ in rows])
    totals: dict[str, list[int]] = {}
    for start, end, name in rows:
        calls, time_ns = totals.get(name, [0, 0])
        totals[name] = [calls + 1, time_ns + (end - start)]
    ranked = sorted(totals.items(), key=lambda item: item[1][1], reverse=True)
    top = [
        {"name": name, "calls": calls, "time_ns": time_ns, "time_s": time_ns / 1e9}
        for name, (calls, time_ns) in ranked[:8]
    ]
    return {
        "num_kernels": len(rows),
        "span_ns": span_ns,
        "busy_ns": busy_ns,
        "gpu_idle_ns": idle_ns,
        "span_s": span_ns / 1e9,
        "busy_s": busy_ns / 1e9,
        "gpu_idle_s": idle_ns / 1e9,
        "idle_fraction_of_span": (idle_ns / span_ns) if span_ns else None,
        "top_kernels": top,
    }


def summarize_nsys_sqlite(path: str | Path) -> dict:
    """Read kernel start/end stamps from an ``nsys export --type sqlite`` file."""
    conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        rows = _kernel_rows(conn)
    finally:
        conn.close()
    return summarize_kernel_rows(rows)


def summarize_ncu_csv(text: str) -> dict:
    """Read an ``ncu --csv`` table, including the unit row under the header."""
    lines = text.splitlines()
    header_at = next((i for i, line in enumerate(lines) if _is_ncu_header(line)), None)
    if header_at is None:
        raise ValueError("ncu CSV has no Kernel Name header")
    reader = csv.DictReader(lines[header_at:])
    parsed = [
        row
        for row in reader
        if any((value or "").strip() for value in row.values() if value is not None)
    ]
    if not parsed:
        raise ValueError("ncu CSV has no kernel rows")
    units = _take_units_row(parsed)
    kernels = [_ncu_kernel(row, units) for row in parsed if (row.get("Kernel Name") or "").strip()]
    return {"num_kernels": len(kernels), "units": units, "kernels": kernels}


def _is_ncu_header(line: str) -> bool:
    return "Kernel Name" in line and "," in line


def _take_units_row(rows: list[dict[str, str | None]]) -> dict[str, str]:
    """ncu prints unit names in the first data row. Drop that row if it is one."""
    first = rows[0]
    values = [
        str(value).strip() for key, value in first.items() if key not in _SKIP_NCU_COLUMNS and value
    ]
    if not values or not all(_looks_like_unit(value) for value in values):
        return {}
    rows.pop(0)
    if not rows:
        raise ValueError("ncu CSV unit row was not followed by a kernel")
    return {
        key: str(value).strip()
        for key, value in first.items()
        if key not in _SKIP_NCU_COLUMNS and value
    }


def _looks_like_unit(value: str) -> bool:
    if value in _DURATION_TO_SECONDS or value in {"byte", "bytes", "%", "sector", "n/a"}:
        return True
    return value.replace("/", "").isalpha()


def _ncu_kernel(row: dict[str, str | None], units: dict[str, str]) -> dict:
    metrics: dict[str, float | str | None] = {}
    for key, value in row.items():
        if key is None or key in _SKIP_NCU_COLUMNS:
            continue
        metrics[key] = _metric_value(value)
    kernel: dict = {"name": str(row.get("Kernel Name")).strip(), "metrics": metrics}
    bandwidth = _achieved_bandwidth(metrics, units)
    if bandwidth is not None:
        kernel["achieved_bandwidth_bytes_per_s"] = bandwidth
    return kernel


def _metric_value(value: str | None) -> float | str | None:
    if value is None:
        return None
    text = value.strip()
    if text == "" or text.lower() == "n/a":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def _achieved_bandwidth(metrics: dict, units: dict[str, str]) -> float | None:
    """Bytes moved over DRAM divided by the kernel duration, when units are known."""
    read = metrics.get("dram__bytes_read.sum")
    write = metrics.get("dram__bytes_write.sum")
    duration = metrics.get("gpu__time_duration.avg")
    duration_unit = units.get("gpu__time_duration.avg")
    if not all(isinstance(value, float) for value in (read, write, duration)):
        return None
    if duration_unit not in _DURATION_TO_SECONDS or duration == 0:
        return None
    seconds = float(duration) * _DURATION_TO_SECONDS[duration_unit]
    return (float(read) + float(write)) / seconds


def _kernel_rows(conn: sqlite3.Connection) -> list[tuple[int, int, str]]:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    table = next((name for name in _KERNEL_TABLES if name in tables), None)
    if table is None:
        raise ValueError(f"nsys sqlite has no kernel table; saw {sorted(tables)[:12]}")
    columns = _columns(conn, table)
    if "start" not in columns or "end" not in columns:
        raise ValueError(f"{table} columns have no start/end: {columns}")
    name_column = next((name for name in _NAME_COLUMNS if name in columns), None)
    if name_column is None:
        raise ValueError(f"{table} columns have no kernel name: {columns}")
    strings = _string_map(conn, tables)
    rows: list[tuple[int, int, str]] = []
    query = f"SELECT start, end, {name_column} FROM {table}"
    for start, end, name in conn.execute(query):
        if start is None or end is None:
            continue
        rows.append((int(start), int(end), _kernel_name(name, strings)))
    if not rows:
        raise ValueError(f"{table} is empty")
    return rows


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def _string_map(conn: sqlite3.Connection, tables: set[str]) -> dict[int, str]:
    if "StringIds" not in tables:
        return {}
    columns = _columns(conn, "StringIds")
    id_column = "id" if "id" in columns else columns[0]
    value_column = "value" if "value" in columns else columns[1]
    return {
        int(row[0]): str(row[1])
        for row in conn.execute(f"SELECT {id_column}, {value_column} FROM StringIds")
    }


def _kernel_name(name, strings: dict[int, str]) -> str:
    if isinstance(name, int):
        return strings.get(name, str(name))
    if name is None:
        return "unknown"
    return str(name)


def write_summary(
    kind: str,
    body: dict,
    output_dir: str | Path,
    *,
    tool_version: str,
    command: str,
    target: dict | None,
) -> Path:
    """Write ``<utc-date>_p2-5-<kind>.json`` next to the other phase 2 results."""
    env = collect_env_info()
    payload = {
        "record": "profile",
        "tool": kind,
        "tool_version": tool_version,
        "command": command,
        "target": target,
        "env_info": env,
        "git_commit": env.get("git_commit"),
        "summary": body,
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now(UTC).strftime('%Y-%m-%d')}_p2-5-{kind}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main(argv: list[str] | None = None) -> None:
    """Write one summary JSON from an nsys sqlite file or an ncu CSV file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=("nsys", "ncu"))
    parser.add_argument("--sqlite", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results/phase2"))
    parser.add_argument("--tool-version", default="unknown")
    parser.add_argument("--command", default="")
    parser.add_argument("--target-json", type=Path)
    args = parser.parse_args(argv)
    if args.kind == "nsys":
        if args.sqlite is None:
            raise SystemExit("nsys summary needs --sqlite")
        body = summarize_nsys_sqlite(args.sqlite)
    else:
        if args.csv is None:
            raise SystemExit("ncu summary needs --csv")
        body = summarize_ncu_csv(args.csv.read_text())
    target = json.loads(args.target_json.read_text()) if args.target_json else None
    path = write_summary(
        args.kind,
        body,
        args.output_dir,
        tool_version=args.tool_version,
        command=args.command,
        target=target,
    )
    print(path)


if __name__ == "__main__":
    main()
