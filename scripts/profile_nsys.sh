#!/usr/bin/env bash
# Timeline of one naive-engine request.
#
# torch.profiler folds launch gaps into the op that waited on them. Nsight
# Systems keeps those gaps, which is what spec Section 11 asks for: GPU idle
# from CPU overhead and synchronization.
#
# Usage (from the repo root):
#   bash scripts/profile_nsys.sh
#   bash scripts/profile_nsys.sh --input-len 64 --output-len 32
#
# nsys is not a project dependency. Point NSYS at the binary, or unpack the
# CUDA redistributable (no root) and leave it under $HOME/opt:
#   mkdir -p "$HOME/opt/nsight-src"
#   wget -O "$HOME/opt/nsight-src/nsys.tar.xz" \
#     https://developer.download.nvidia.com/compute/cuda/redist/nsight_systems/linux-x86_64/nsight_systems-linux-x86_64-2026.3.2.476-archive.tar.xz
#   mkdir -p "$HOME/opt/nsight-systems"
#   tar -C "$HOME/opt/nsight-systems" -xJf "$HOME/opt/nsight-src/nsys.tar.xz"
#
# The raw .nsys-rep and .sqlite stay in profiles/ (gitignored). A small JSON
# summary is written under docs/results/phase2/. On WSL the script sets
# CuptiUseRawGpuTimestamps=false, which Nsight documents as the safe clock.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NSYS_BIN="$(bash "$ROOT/scripts/profile_tool.sh" nsys)"
PY="$(bash "$ROOT/scripts/profile_tool.sh" python)"
mkdir -p profiles/nsys

# WSL's default GPU clock conversion is not reliable. Append once.
NSYS_CONFIG="$("$NSYS_BIN" -z)"
mkdir -p "$(dirname "$NSYS_CONFIG")"
if ! grep -q '^CuptiUseRawGpuTimestamps=false$' "$NSYS_CONFIG" 2>/dev/null; then
  echo 'CuptiUseRawGpuTimestamps=false' >>"$NSYS_CONFIG"
fi

REP="profiles/nsys/naive"
rm -f "${REP}.nsys-rep" "${REP}.sqlite" profiles/last_target.json
export HF_HUB_OFFLINE=1
cmd=(
  "$NSYS_BIN" profile
  --force-overwrite=true
  --trace=cuda
  --capture-range=cudaProfilerApi
  --capture-range-end=stop
  --output="$REP"
  "$PY" -m scripts.profile_target "$@"
)
export PROFILE_COMMAND="${cmd[*]}"
"${cmd[@]}"
"$NSYS_BIN" export --type sqlite --force-overwrite=true --output "${REP}.sqlite" "${REP}.nsys-rep"
TOOL_VERSION="$("$NSYS_BIN" --version | head -n 1)"
"$PY" -m scripts.profile_summary nsys \
  --sqlite "${REP}.sqlite" \
  --output-dir docs/results/phase2 \
  --tool-version "$TOOL_VERSION" \
  --command "$PROFILE_COMMAND" \
  --target-json profiles/last_target.json
