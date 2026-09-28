#!/usr/bin/env bash
# Download Llama checkpoints (safetensors, configs, tokenizer) into models/<name>.
#
# Why a script: weights are never committed (spec Section 3), so every machine fetches the same
# files the same way. Llama repos are gated: accept the license on huggingface.co first and
# export HF_TOKEN.
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash scripts/download_models.sh                       # dev model (Llama-3.2-1B-Instruct)
#   bash scripts/download_models.sh dev-spec              # 3B target + 1B draft
#   bash scripts/download_models.sh final                 # 8B target + 1B draft
#   bash scripts/download_models.sh meta-llama/<repo> ... # explicit repo ids
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "error: HF_TOKEN is not set. Accept the Llama license on Hugging Face, then:" >&2
  echo "  export HF_TOKEN=hf_..." >&2
  exit 1
fi

MODELS_DIR="${MODELS_DIR:-models}"

case "${1:-dev}" in
  dev) repos=("meta-llama/Llama-3.2-1B-Instruct") ;;
  dev-spec) repos=("meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.2-1B-Instruct") ;;
  final) repos=("meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.2-1B-Instruct") ;;
  *) repos=("$@") ;;
esac

for repo in "${repos[@]}"; do
  dest="${MODELS_DIR}/${repo#*/}"
  echo "==> ${repo} -> ${dest}"
  # Only safetensors, configs, and tokenizer files; skip the original/ consolidated .pth copy.
  uv run --with huggingface_hub hf download "${repo}" \
    --include "*.safetensors" "*.json" "tokenizer*" \
    --local-dir "${dest}" \
    --token "${HF_TOKEN}"
done
