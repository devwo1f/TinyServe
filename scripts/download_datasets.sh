#!/usr/bin/env bash
# Download benchmark and evaluation datasets into data/ (never committed, spec Section 5).
#
#   ShareGPT (V3 unfiltered cleaned split)  -> data/sharegpt/   serving workloads
#   WikiText-2 raw (test split)             -> data/wikitext/   perplexity (Phase 9)
#   HumanEval prompts (official JSONL, not the HF parquet) -> data/humaneval/  code workload
#
# Filtering and fixed-seed subsetting happen in bench/datasets.py, not here, so the raw files
# stay exactly as published.
#
# Usage: bash scripts/download_datasets.sh [sharegpt|wikitext|humaneval ...]   (default: all)
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
token_args=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  token_args=(--token "${HF_TOKEN}")
fi

hf_dataset() {
  local repo="$1" dest="$2"
  shift 2
  echo "==> ${repo} -> ${dest}"
  uv run --with huggingface_hub hf download "${repo}" --repo-type dataset \
    --local-dir "${dest}" "${token_args[@]}" "$@"
}

targets=("$@")
if [[ ${#targets[@]} -eq 0 ]]; then
  targets=(sharegpt wikitext humaneval)
fi

for t in "${targets[@]}"; do
  case "${t}" in
    sharegpt)
      hf_dataset anon8231489123/ShareGPT_Vicuna_unfiltered "${DATA_DIR}/sharegpt" \
        --include "ShareGPT_V3_unfiltered_cleaned_split.json" ;;
    wikitext)
      hf_dataset Salesforce/wikitext "${DATA_DIR}/wikitext" \
        --include "wikitext-2-raw-v1/*" ;;
    humaneval)
      # The Hugging Face copy is parquet (D-009). The official release is JSONL,
      # which bench/datasets.py reads without an extra dependency.
      dest="${DATA_DIR}/humaneval"
      mkdir -p "${dest}"
      echo "==> openai/human-eval JSONL -> ${dest}"
      curl -fsSL -o "${dest}/HumanEval.jsonl.gz" \
        https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz
      gzip -dc "${dest}/HumanEval.jsonl.gz" > "${dest}/humaneval.jsonl" ;;
    *)
      echo "error: unknown dataset '${t}' (expected sharegpt, wikitext, humaneval)" >&2
      exit 1 ;;
  esac
done
