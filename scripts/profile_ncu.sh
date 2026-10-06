#!/usr/bin/env bash
# Bandwidth and occupancy for a few kernels of one naive-engine request.
#
# Nsight Compute replays kernels, so this does not time the whole request.
# It samples the first N launches inside the CUDA profiler range (after the
# warmup request). The default range is one decode-shaped token so those
# launches are matrix-vector products, which is the memory-bound step.
#
# Usage (from the repo root):
#   bash scripts/profile_ncu.sh
#   NCU_LAUNCH_COUNT=4 bash scripts/profile_ncu.sh --input-len 1 --output-len 2
#
# ncu is not a project dependency. Point NCU at the binary, or unpack:
#   mkdir -p "$HOME/opt/nsight-src" "$HOME/opt/nsight-compute"
#   wget -O "$HOME/opt/nsight-src/ncu.tar.xz" \
#     https://developer.download.nvidia.com/compute/cuda/redist/nsight_compute/linux-x86_64/nsight_compute-linux-x86_64-2026.3.1.2-archive.tar.xz
#   tar -C "$HOME/opt/nsight-compute" -xJf "$HOME/opt/nsight-src/ncu.tar.xz"
#
# On WSL the counters are enforced by the Windows driver, not by Linux.
# NVIDIA App (driver R610 and newer): System > Advanced > Developer >
# Manage GPU Performance Counters > Allow access to all users.
# Older drivers: NVIDIA Control Panel > Desktop > Enable Developer Settings,
# then Developer > Manage GPU Performance Counters > allow all users.
# `sudo` inside WSL does not grant this. Until it is enabled, ncu exits with
# ERR_NVGPUCTRPERM and this script does not write a result file.
# The raw .ncu-rep stays in profiles/ (gitignored). The CSV summary is the
# file under docs/results/phase2/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NCU_BIN="$(bash "$ROOT/scripts/profile_tool.sh" ncu)"
PY="$(bash "$ROOT/scripts/profile_tool.sh" python)"
LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-8}"
mkdir -p profiles/ncu

REP="profiles/ncu/naive"
rm -f "${REP}.ncu-rep" "${REP}.csv" profiles/last_target.json
export HF_HUB_OFFLINE=1
# One prompt token, so the sampled launches are decode-shaped. Skip the
# default when the caller already passed that flag.
args=("$@")
if [[ " ${args[*]} " != *" --input-len "* ]]; then
  args+=(--input-len 1)
fi
if [[ " ${args[*]} " != *" --output-len "* ]]; then
  args+=(--output-len 2)
fi

cmd=(
  "$NCU_BIN"
  --profile-from-start off
  --target-processes all
  --replay-mode kernel
  --launch-count "$LAUNCH_COUNT"
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.avg,sm__warps_active.avg.pct_of_peak_sustained_active
  --force-overwrite
  --export "$REP"
  "$PY" -m scripts.profile_target "${args[@]}"
)
export PROFILE_COMMAND="${cmd[*]}"
"${cmd[@]}"
"$NCU_BIN" --import "${REP}.ncu-rep" --csv >"${REP}.csv"
TOOL_VERSION="$("$NCU_BIN" --version | head -n 1)"
"$PY" -m scripts.profile_summary ncu \
  --csv "${REP}.csv" \
  --output-dir docs/results/phase2 \
  --tool-version "$TOOL_VERSION" \
  --command "$PROFILE_COMMAND" \
  --target-json profiles/last_target.json
