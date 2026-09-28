# TinyServe: End-to-End Project Context

> Handoff document for any human or AI picking up this project. It is updated in the same commit as every change. If it disagrees with the code, the code wins; fix this file.
> The full specification is `docs/SPEC.md`. The spec is the source of truth for *what* to build; this file describes *where the project is now*. Reading order for a new agent: this file, `docs/SPEC.md` Section 3, the latest `docs/PROGRESS.md` entries, all of `docs/DECISIONS.md`.

---

## 1. Project purpose

TinyServe is a small, readable LLM inference serving engine built from scratch in Python, PyTorch, and Triton. It serves a Llama-architecture model on a single GPU through an OpenAI-compatible streaming API. The owner (Abhayraj Singh, GitHub `devwo1f`) is targeting ML inference / ML systems roles; the project must demonstrate first-hand understanding of KV caching, paged memory, attention kernels, continuous batching, serving, profiling, speculative decoding, and quantization.

Priorities, in order:
1. **Correctness**: outputs match Hugging Face within defined tolerances.
2. **Readability**: any module understandable in under 15 minutes; shape comments; docstrings explain *why*.
3. **Measurability**: every optimization benchmarked before/after, raw results committed.
4. Target 50 to 80 percent of vLLM throughput on the same hardware; explaining the gap is part of the deliverable.

Non-goals: multi-GPU, non-Llama architectures, beam search, LoRA, guided decoding, production hardening, beating vLLM.

Honesty constraints (non-negotiable): no fabricated or hand-edited numbers; no novelty claims; no copied code from other engines (conceptual borrowing recorded in `docs/REFERENCES.md`); negative results reported as-is; loosened tolerances documented in `docs/DECISIONS.md`.

## 2. Current status

- **Phase:** 0 (Repository and Environment)
- **Last completed task:** P0.4 config
- **In progress:** P0.5 CI
- **Review gates passed:** none yet
- **GitHub:** https://github.com/devwo1f/TinyServe (public)

## 3. Phase roadmap

| Phase | Content | Status |
|---|---|---|
| 0 | Skeleton, docs, env scripts, config, CI | in progress |
| 1 | From-scratch Llama (RoPE w/ Llama 3 scaling, GQA, SwiGLU), safetensors loading, sampler, naive engine, HF parity | not started |
| 2 | Benchmark + profiling harness, HF and vLLM baselines | not started |
| 3 | Paged KV cache, block manager, KV store, reference paged attention, prefix cache | not started |
| 4 | Continuous batching scheduler, chunked prefill, preemption, engine step loop | not started |
| 5 | Triton kernels: fused add+RMSNorm, RoPE, paged decode attention (split-K and prefill are stretch) | not started |
| 6 | CUDA graphs for decode | not started |
| 7 | OpenAI-compatible FastAPI SSE server, engine thread, SLA-aware admission control | not started |
| 8 | Speculative decoding (draft model, rejection sampling, KV rollback) | not started |
| 9 | INT8 weight-only quantization + W8A16 Triton GEMM, perplexity eval | not started |
| 10 | Research study: speculation-length policies under load (reproduction, not novel) | not started |
| 11 | README, technical write-up, upstream contribution (human-led) | not started |

## 4. Architecture and data flow (target design)

```
Client --HTTP/SSE--> server/api.py --> server/admission.py --> engine/engine.py
engine.py --> engine/scheduler.py --> kv/block_manager.py, kv/prefix_cache.py
engine.py --> engine/model_runner.py --> model/llama.py --> kernels/*, kv/cache.py
model_runner.py --> engine/cuda_graphs.py
engine.py --> spec/* (speculative decoding), engine/sampler.py
```

Request lifecycle: API tokenizes and creates a `Sequence` -> admission control admits / queues with deadline / rejects (503) -> scheduler builds a `ScheduledBatch` per step (decodes first, then prefill chunks, within token budget and free KV blocks) -> block manager allocates blocks, prefix cache reuses full blocks -> model runner flattens batch into 1-D tensors and runs the model (decode-only batches use CUDA graphs) -> sampler (or drafter + verifier) produces tokens -> tokens streamed back; finished sequences release blocks.

Engine loop runs in a background thread (Phase 7), talking to the async API through thread-safe queues.

## 5. Key contracts (from spec Section 9; changing them needs a DECISIONS.md entry)

- `SamplingParams(temperature=1.0 [0 = greedy], top_p=1.0, top_k=-1, max_tokens=256, seed=None, stop_token_ids=[])`
- `SequenceStatus`: WAITING, RUNNING, PREEMPTED, FINISHED
- `Sequence`: `seq_id, prompt_token_ids, output_token_ids, sampling_params, status, num_computed_tokens, block_table, arrival_time, first_token_time, num_draft_tokens_proposed, num_draft_tokens_accepted`
- KV cache per layer: `k_cache/v_cache[layer]: [num_blocks, block_size, num_kv_heads, head_dim]`, default `block_size=16`
- Token at logical position `p` lives in block `block_table[p // block_size]`, offset `p % block_size`; `slot_mapping[i] = block_id * block_size + offset`
- `num_blocks` from memory profiling: `kv_bytes = total_mem * gpu_memory_utilization - weights - peak_activations`
- Prefix cache: only full blocks; hash = hash(parent_hash, tuple(block_token_ids)); LRU evictable pool; always recompute the last prompt token
- `ScheduledSeq(seq, num_new_tokens, is_prefill)`, `ScheduledBatch(seqs, num_batched_tokens, preempted)`
- Model runner inputs: `input_ids, positions, slot_mapping [num_tokens] int64`; `query_start_loc [num_seqs+1] int32`; `seq_lens [num_seqs] int32`; `block_tables [num_seqs, max_blocks] int32`; `logits_indices [num_seqs] int64`

## 6. Repo map (what exists right now)

| Path | Purpose |
|---|---|
| `CONTEXT.md` | This file: end-to-end project context, updated every change |
| `CLAUDE.md`, `AGENTS.md` | Identical agent instructions for Claude Code / Codex (spec Appendix A plus "read CONTEXT.md first" and the git workflow) |
| `docs/SPEC.md` | Full build specification (source of truth) |
| `docs/PROGRESS.md` | Session log, newest first (Appendix B template); human writes review-gate confirmations here |
| `docs/DECISIONS.md` | Decisions `D-<n>`, open questions `Q-<n>`, loosened tolerances |
| `docs/REFERENCES.md` | Papers/projects consulted and a "conceptual borrowing" table |
| `docs/learn/README.md` | Learning-note format, review gates, index of notes |
| `docs/learn/p0-setup.md` | Phase 0 learning note (updated through P0.5) |
| `docs/results/` | Raw benchmark JSONL (script-written only), README with rules |
| `docs/writeup/` | Final technical write-up (Phase 11) |
| `.cursor/rules/tinyserve-workflow.mdc` | Always-on Cursor rule: read CONTEXT.md first, update it each change, commit format, push after every commit |
| `.gitignore` | Excludes weights, datasets, secrets, profiler outputs, large results |
| `.gitattributes` | Forces LF line endings (scripts run on Linux/WSL/CI) |
| `pyproject.toml` | uv project (Python 3.11 only). Base deps: `numpy`. Torch `2.14.0` via mutually exclusive extras `cu130` (GPU) / `cpu` (CI) from the official PyTorch indexes. Dev group `pytest` + `ruff`. Pytest markers `gpu`/`slow` (`--strict-markers`), `pythonpath = ["."]`. Ruff: Python files only, line length 100 |
| `scripts/env_info.py` | `collect_env_info() -> dict` with fixed `FIELDS` (timestamp, git commit/dirty, python, platform, cpu, torch, torch_cuda, cudnn, triton, cuda_available, gpu_count/name/memory/compute capability, driver). Missing items are None. `--json` flag. Embedded in every result file |
| `scripts/download_models.sh` | `bash scripts/download_models.sh [dev / dev-spec / final / <repo ids>]` into `models/<name>` (safetensors, json, tokenizer). Requires `HF_TOKEN`. Uses `uv run --with huggingface_hub hf download` |
| `scripts/download_datasets.sh` | `bash scripts/download_datasets.sh [sharegpt wikitext humaneval]` into `data/` (raw files; filtering happens in `bench/datasets.py`) |
| `uv.lock`, `.python-version` | Locked dependency set; Python pin `3.11` |
| `README.md` | Short public README (no numbers until result files exist) |
| `tinyserve/` | Main package. `__init__.py` holds `__version__`. Subpackages (each only an `__init__.py` docstring so far): `model/`, `kv/`, `engine/`, `kernels/`, `spec/`, `quant/`, `server/`. Module files from spec Section 8 are created by the task that implements them, not as empty stubs. |
| `tinyserve/config.py` | Torch-free settings. `TinyServeConfig` has sections `model` (model path, tokenizer, dtype `auto`/float32/float16/bfloat16, device, max_model_len, seed), `cache` (block_size 16, gpu_memory_utilization 0.9, memory_safety_margin_gib, num_gpu_blocks_override, enable_prefix_caching), `scheduler` (max_num_batched_tokens 2048, max_num_seqs 64, enable_chunked_prefill), `speculative` (enabled, draft_model, num_speculative_tokens, policy, batch_threshold), `server` (host, port, admission_policy fifo/reject/deadline, TTFT/TPOT SLOs), `benchmark` (workload, num_requests, request_rate, warmup, repeats, seed, ignore_eos, output_dir). API: `apply_overrides(cfg, {"cache.block_size": "32"})`, `add_config_args(parser)` adds `--section.field` flags, `config_from_args(args)`, `cfg.to_dict()` |
| `bench/`, `bench/microbench/` | Benchmark package (empty until Phase 2) |
| `eval/` | Parity and perplexity evaluation (empty until Phase 1/9) |
| `tests/fixtures/tiny_llama.json` | Tiny random Llama config in HF `config.json` format: 2 layers, hidden 64, 4 Q heads, 2 KV heads, head_dim 16, vocab 256, intermediate 128, Llama 3 `rope_scaling` |
| `tests/unit/test_skeleton.py` | Smoke tests: package imports, fixture matches spec Section 6 |
| `tests/unit/test_env_info.py` | env_info returns all fields; `--json` CLI works |
| `tests/unit/test_config.py` | Config defaults, overrides, validation, CLI flags |
| `tests/gpu/test_env_info_gpu.py` | (`gpu`) GPU fields are populated |

## 7. Environment and hardware

- Dev machine: Windows 11, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, compute capability 8.9), driver 595.97. No `nvcc` on PATH. torch 2.14.0+cu130 works natively (CUDA available); Triton is not installed on native Windows.
- Tools: git 2.52, GitHub CLI 2.87 (logged in as `devwo1f`), uv 0.11. System Python is 3.13; project pins Python 3.11 via uv.
- Implications:
  - Llama-3.2-1B-Instruct in bf16 (~2.5 GB) fits: Phase 1 to 7 development can be local.
  - 3B target + 1B draft (~9 GB bf16) does not fit in 8 GB: needs INT8 target or a cloud GPU (open question).
  - Final Llama-3.1-8B benchmarks need an A100/H100 (cloud).
  - Triton has no official native-Windows support and vLLM needs Linux: WSL2 recommended from Phase 1 (open question).
- CPU-only unit tests use a tiny random Llama config (`tests/fixtures/tiny_llama.json`, added in P0.1).

## 8. How to run and test

```bash
uv sync --extra cu130            # GPU machine: .venv with Python 3.11, torch 2.14.0+cu130, dev tools
uv sync --extra cpu              # CPU-only machine / CI
uv run python scripts/env_info.py
uv run ruff check .              # lint
uv run ruff format --check .     # formatting
uv run pytest -m "not gpu"       # CPU tests (always)
uv run pytest -m gpu             # GPU tests (when a CUDA GPU is available)
```

Note: on this Windows machine uv warns it cannot hardlink from its cache (different drive); harmless. Set `UV_LINK_MODE=copy` to silence it.

## 9. Workflow rules

- Branch per task: `p<phase>-<task>-<slug>`. Commit message: `P<phase>.<task>: <short description>`.
- Push after **every** commit. At task end: PR to `main` via `gh pr create`, merge with `gh pr merge --merge --delete-branch` once acceptance criteria (and CI, from P0.5) pass.
- Update this file in the same commit as every change, including a change-log line.
- Session end: `docs/PROGRESS.md` entry + `docs/learn/` note for finished tasks.
- Do not start a new phase until the human confirms the review gate in `docs/PROGRESS.md`.

## 10. Decisions summary

Full entries are in `docs/DECISIONS.md`.

- D-001: Python 3.11 via uv; not an installable package yet (pytest `pythonpath = ["."]`).
- D-002: ruff only checks Python files (keeps spec Markdown snippets untouched).
- D-003: branch per task, push every commit, merge by PR with merge commits; LF line endings.
- D-004: torch 2.14.0 pinned, `cu130`/`cpu` extras; numpy base dependency; huggingface_hub is only used through `uv run --with`.
- D-005: `scripts/` is an importable package (bench code embeds `collect_env_info()`).
- D-006: config is torch-free dataclasses with dotted overrides; `dtype="auto"` is resolved by model code in Phase 1.
- Repo is public on GitHub (human choice). Scope of the first execution run: Phase 0 only, then stop for human review.

## 11. Open questions / known issues

- Q-001: WSL2 vs native Windows for Triton and vLLM work (decide before Phase 1; WSL2 recommended).
- Q-002: how to fit 3B target + 1B draft for speculative decoding on an 8 GB GPU (INT8 target vs cloud GPU; decide before Phase 8).
- The human must accept the Llama license on Hugging Face and provide `HF_TOKEN` before Phase 1.

## 12. Next steps

1. P0.5 GitHub Actions CI (ruff + `pytest -m "not gpu"` with `uv sync --extra cpu`). Then stop for human review.

## 13. Change log

- 2026-09-28 bootstrap: git repo, public GitHub remote, CONTEXT.md, Cursor workflow rule, .gitignore, .gitattributes.
- 2026-09-28 P0.1: package skeleton, pyproject/uv (Python 3.11), ruff + pytest config, tiny_llama.json fixture, smoke tests.
- 2026-09-28 P0.2: spec moved to docs/SPEC.md; PROGRESS, DECISIONS (D-001..D-003, Q-001, Q-002), REFERENCES, learn notes, results/writeup dirs, CLAUDE.md, AGENTS.md.
- 2026-09-28 P0.3: scripts/env_info.py, download_models.sh, download_datasets.sh; torch 2.14.0 (cu130/cpu extras) and numpy.
- 2026-09-28 P0.4: tinyserve/config.py (section dataclasses, validation, dotted overrides, CLI flags) + tests.
