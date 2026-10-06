#!/usr/bin/env bash
# Print the nsys, ncu, or project-python path the profiling wrappers should use.
# Exists so both wrappers resolve tools the same way. Not a profiler itself.
set -euo pipefail

name="${1:-}"
case "$name" in
  python)
    if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
      printf '%s\n' "$PYTHON"
      exit 0
    fi
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    if [[ -x "$root/.venv/bin/python" ]]; then
      printf '%s\n' "$root/.venv/bin/python"
      exit 0
    fi
    echo "error: no project python at .venv/bin/python (uv sync --extra cu130)" >&2
    exit 1
    ;;
  nsys) env_var="${NSYS:-}" ;;
  ncu) env_var="${NCU:-}" ;;
  *)
    echo "usage: profile_tool.sh {nsys|ncu|python}" >&2
    exit 1
    ;;
esac

if [[ -n "$env_var" && -x "$env_var" ]]; then
  printf '%s\n' "$env_var"
  exit 0
fi
if command -v "$name" >/dev/null 2>&1; then
  command -v "$name"
  exit 0
fi

found=""
for root in "$HOME/opt/nsight-systems" "$HOME/opt/nsight-compute" "$HOME/opt"; do
  if [[ -d "$root" ]]; then
    while IFS= read -r path; do
      if [[ "$path" == *t210* || "$path" == *aarch64* || "$path" == *sbsa* ]]; then
        continue
      fi
      # The x64 host binary, not a copy buried in a foreign target tree.
      if [[ -z "$found" || ${#path} -lt ${#found} ]]; then
        found="$path"
      fi
    done < <(find "$root" -type f -name "$name" -perm -u+x 2>/dev/null || true)
    if [[ -n "$found" ]]; then
      printf '%s\n' "$found"
      exit 0
    fi
  fi
done

echo "error: $name not found. Set ${name^^} or unpack the archive under \$HOME/opt. See scripts/profile_${name}.sh." >&2
exit 1
