# Benchmark Results

Raw benchmark output lives here as `docs/results/<phase>/<date>_<name>.jsonl` (spec Section 11). Each file contains the config, `scripts/env_info.py` output, git commit hash, per-request records, and summary metrics.

Rules:
- Files here are written only by scripts in this repo. Never edit them by hand.
- Plots are generated only by `bench/plots.py` from these files.
- Keep individual files under 5 MB; larger raw artifacts stay local (see `.gitignore`).
