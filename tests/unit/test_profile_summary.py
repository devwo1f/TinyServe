"""Nsight summary math. No GPU and no nsys/ncu binary required."""

import sqlite3
import subprocess
from pathlib import Path

import pytest

from scripts.profile_summary import coverage_ns, summarize_ncu_csv, summarize_nsys_sqlite

ROOT = Path(__file__).resolve().parents[2]


def test_idle_gap_between_two_kernels():
    assert coverage_ns([(0, 10), (20, 30)]) == (30, 20, 10)


def test_overlapping_kernels_count_once():
    assert coverage_ns([(0, 10), (5, 12), (20, 25)]) == (25, 17, 8)


def test_sqlite_kernel_names_join_string_ids(tmp_path: Path):
    db = tmp_path / "naive.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
        "(start INTEGER, end INTEGER, demangledName INTEGER)"
    )
    conn.executemany("INSERT INTO StringIds VALUES (?, ?)", [(1, "gemv_a"), (2, "gemv_b")])
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?)",
        [(0, 10, 1), (20, 30, 2)],
    )
    conn.commit()
    conn.close()

    body = summarize_nsys_sqlite(db)
    assert body["num_kernels"] == 2
    assert body["gpu_idle_ns"] == 10
    assert body["busy_ns"] == 20
    assert body["top_kernels"][0]["name"] == "gemv_a"


def test_ncu_csv_bandwidth_uses_the_unit_row():
    text = """==PROF== connected
"ID","Kernel Name","dram__bytes_read.sum","dram__bytes_write.sum","gpu__time_duration.avg"
"","","byte","byte","ns"
"1","nvjet_gemv","1000","500","100"
"""
    body = summarize_ncu_csv(text)
    assert body["num_kernels"] == 1
    assert body["units"]["gpu__time_duration.avg"] == "ns"
    # 1500 bytes / 100 ns. The division is binary floating point.
    assert body["kernels"][0]["achieved_bandwidth_bytes_per_s"] == pytest.approx(1.5e10)


def test_profile_scripts_are_valid_bash():
    for name in ("profile_nsys.sh", "profile_ncu.sh", "profile_tool.sh"):
        script = ROOT / "scripts" / name
        text = script.read_text()
        assert "scripts.profile_target" in text or name == "profile_tool.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
