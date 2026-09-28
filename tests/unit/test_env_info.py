"""Tests for scripts/env_info.py.

Result files embed this output, so it must always produce every field, on any machine.
"""

import json
import subprocess
import sys

from scripts.env_info import FIELDS, REPO_ROOT, collect_env_info


def test_collect_has_all_fields():
    info = collect_env_info()
    assert list(info) == FIELDS
    assert info["python"] == sys.version.split()[0]
    assert info["timestamp_utc"] is not None


def test_json_cli_roundtrip():
    out = subprocess.run(
        [sys.executable, "scripts/env_info.py", "--json"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    assert set(json.loads(out.stdout)) == set(FIELDS)
