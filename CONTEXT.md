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

- **Phase:** 3 (Paged KV cache)
- **Last completed task:** P3.1 KV cache allocation and memory profiling
- **In progress:** P3.2 block manager. Dev environment is WSL2 (D-007). Llama 3.2 access is approved and `Llama-3.2-1B-Instruct` is downloaded locally. Human asked to start Phase 3 on 2026-10-06. Nsight Compute counters are still blocked on the Windows host; that does not block paging.
- **Working copy:** `~/TinyServe` inside WSL2 Ubuntu 24.04 (user `abhay`), opened in Cursor via the WSL remote. Do not develop in the old `D:\Projects\TinyServe` Windows copy.
- **Review gates passed:** Phase 0, Phase 1, and Phase 2 (human asked to start the next phase)
- **GitHub:** https://github.com/devwo1f/TinyServe (public)

## 3. Phase roadmap

| Phase | Content | Status |
|---|---|---|
| 0 | Skeleton, docs, env scripts, config, CI | done (human asked to start Phase 1) |
| 1 | From-scratch Llama (RoPE w/ Llama 3 scaling, GQA, SwiGLU), safetensors loading, sampler, naive engine, HF parity | done (human asked to start Phase 2) |
| 2 | Benchmark + profiling harness, HF and vLLM baselines | done (human asked to start Phase 3; ncu counters still blocked on the Windows host) |
| 3 | Paged KV cache, block manager, KV store, reference paged attention, prefix cache | in progress (P3.1 done) |
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
| `pyproject.toml` | uv project (Python 3.11 only). Base deps: `numpy`, `transformers==5.17.0` (tokenizer and HF reference). Torch `2.14.0` via mutually exclusive extras `cu130` (GPU) / `cpu` (CI) from the official PyTorch indexes. Dev group `pytest` + `ruff`. Pytest markers `gpu`/`slow` (`--strict-markers`), `pythonpath = ["."]`. Ruff: Python files only, line length 100 |
| `scripts/env_info.py` | `collect_env_info() -> dict` with fixed `FIELDS` (timestamp, git commit/dirty, python, platform, cpu, torch, torch_cuda, cudnn, triton, cuda_available, gpu_count/name/memory/compute capability, driver). Missing items are None. `--json` flag. Embedded in every result file |
| `scripts/download_models.sh` | `bash scripts/download_models.sh [dev / dev-spec / final / <repo ids>]` into `models/<name>`. Requires `HF_TOKEN` in the environment (not passed on the command line). One `--include` flag per glob, or the CLI treats extras as filenames and skips the weights. Excludes `original/` (duplicate .pth). Dev model is already at `models/Llama-3.2-1B-Instruct` (safetensors, not committed) |
| `scripts/download_datasets.sh` | `bash scripts/download_datasets.sh [sharegpt wikitext humaneval]` into `data/` (raw files; filtering happens in `bench/datasets.py`). HumanEval is the official JSONL, not the Hugging Face parquet (D-009). |
| `uv.lock`, `.python-version` | Locked dependency set; Python pin `3.11` |
| `README.md` | Short public README with CI badge (no numbers until result files exist) |
| `.github/workflows/ci.yml` | CI on push to main and PRs: ubuntu, `uv sync --locked --extra cpu`, env_info, ruff check, ruff format --check, `pytest -m "not gpu"` |
| `tinyserve/` | Main package. `__init__.py` holds `__version__`. Subpackages: `model/`, `kv/`, `engine/`, `kernels/`, `spec/`, `quant/`, `server/`. Module files from spec Section 8 are created by the task that implements them. |
| `tinyserve/model/tokenizer.py` | `Tokenizer` wraps HF: `from_pretrained`, `encode` (no special tokens by default), `decode`, `apply_chat_template`, `eos_token_id` (`<\|eot_id\|>` for Llama 3 Instruct), `bos_token_id`. `IncrementalDetokenizer.add` decodes the full id list and holds back a trailing U+FFFD so a character split across tokens is not streamed as a replacement box. `finish` flushes the tail. |
| `tinyserve/model/rope.py` | `compute_inv_freq` (`[head_dim/2]`), Llama 3 wavelength scaling via `RopeScaling`, `rotary_cos_sin` (`[batch, seq, head_dim]`), `apply_rotary`. Checked against HF within 1e-5, including positions past 8192. |
| `tinyserve/model/llama.py` | `LlamaModelConfig`, `RMSNorm` (variance in fp32), `Attention` (GQA, SDPA), `MLP` (SwiGLU), `DecoderLayer`, `LlamaModel`, `LlamaForCausalLM`. `ContiguousKVCache` is `[num_layers, batch, max_len, num_kv_heads, head_dim]`. Parameter names match HF. Tied embeddings share `lm_head.weight` with `embed_tokens.weight`. A single new token uses unmasked SDPA (it can see the whole cache). A longer query that is still shorter than the cache uses an explicit causal mask, because `is_causal=True` on this PyTorch does not align a short query to the end of a longer cache. |
| `tinyserve/model/weights.py` | `load_hf_weights` reads safetensors straight onto the target device and dtype. Names already match HF. A missing `lm_head.weight` is allowed only when embeddings are tied. |
| `tinyserve/config.py` | Torch-free settings. `TinyServeConfig` has sections `model` (model path, tokenizer, dtype `auto`/float32/float16/bfloat16, device, max_model_len, seed), `cache` (block_size 16, gpu_memory_utilization 0.9, memory_safety_margin_gib, num_gpu_blocks_override, enable_prefix_caching), `scheduler` (max_num_batched_tokens 2048, max_num_seqs 64, enable_chunked_prefill), `speculative` (enabled, draft_model, num_speculative_tokens, policy, batch_threshold), `server` (host, port, admission_policy fifo/reject/deadline, TTFT/TPOT SLOs), `benchmark` (workload, num_requests, request_rate, warmup, repeats, seed, ignore_eos, output_dir). API: `apply_overrides(cfg, {"cache.block_size": "32"})`, `add_config_args(parser)` adds `--section.field` flags, `config_from_args(args)`, `cfg.to_dict()` |
| `tinyserve/engine/sequence.py` | `SamplingParams` (temperature 0 is greedy, top-k -1 is off), `SequenceStatus`, and `Sequence` with the spec Section 9 fields. |
| `tinyserve/engine/sampler.py` | `sample_token`: temperature, then top-k, then top-p. A per-request `torch.Generator` keeps seeds from sharing the global RNG. |
| `tinyserve/engine/engine.py` | `Engine.generate(prompts, sampling_params)` runs one request at a time on a fresh `ContiguousKVCache`. Stop ids are not emitted. `generate_tokens` is the same loop for callers who already have ids. `GenerationResult` carries `ttft_s`, `e2e_s`, and `itl_s`, taken after the logits for that token are ready. The cache write after the last token is outside those times. |
| `tinyserve/kv/cache.py` | Paged K/V tensors and the startup block count. Budget is `total * utilization - weights - peak activations - safety margin`. `str(CacheProfile)` is the startup line: blocks and token capacity. Tied weights are counted once. |
| `bench/datasets.py` | ShareGPT, code (JSON/JSONL), shared-prefix, and synthetic workloads. Length filter, then a hash-ordered subset for a fixed seed. `prompt_token_ids` is what a benchmark must send. |
| `bench/offline.py` | `run_offline` runs each sample to a fixed length, drops a separate warmup pass, repeats, and writes one Section 11 JSONL (meta, per-request, summary). Summary percentiles come from the median repeat by output throughput. `request_record` / `write_benchmark_jsonl` are the shared row shape. |
| `bench/baselines.py` | `run_hf_generate` times Hugging Face `generate` on the same samples. `profile_generation` writes a `torch.profiler` summary of one TinyServe request. The CLI loads one model at a time. |
| `bench/vllm_offline.py` | Runs the same synthetic set through vLLM's `LLM.generate` (one batched call per repeat) and writes a Section 11 JSONL. Executed with the separate venv, not TinyServe's. |
| `bench/vllm_baseline.md` | Exact install commands, versions, and the run command for that venv (`/home/abhay/venvs/tinyserve-vllm`, vLLM 0.31.0+cu129). |
| `scripts/profile_target.py` | One greedy naive-engine request. Warmup stays outside `cudaProfilerStart` / `Stop`. |
| `scripts/profile_nsys.sh` | Nsight Systems timeline of that request. Writes a small JSON summary; the `.nsys-rep` stays in `profiles/`. |
| `scripts/profile_ncu.sh` | Nsight Compute on the first few launches inside the same profiler range. Needs Windows GPU performance counters. |
| `scripts/profile_summary.py` | Turns an nsys sqlite or an ncu CSV into the result JSON. `coverage_ns` is the idle-gap math. |
| `bench/microbench/` | Kernel microbenchmarks (empty until Phase 5) |
| `eval/` | Parity and perplexity evaluation (empty until Phase 1/9) |
| `tests/fixtures/tiny_llama.json` | Tiny random Llama config in HF `config.json` format: 2 layers, hidden 64, 4 Q heads, 2 KV heads, head_dim 16, vocab 256, intermediate 128, Llama 3 `rope_scaling` |
| `tests/unit/test_skeleton.py` | Smoke tests: package imports, fixture matches spec Section 6 |
| `tests/unit/test_env_info.py` | env_info returns all fields; `--json` CLI works |
| `tests/unit/test_config.py` | Config defaults, overrides, validation, CLI flags |
| `tests/unit/test_tokenizer.py` | Round-trip, chat template, and incremental detokenization. A tiny byte-level BPE always runs. Llama 3.2 tests run only if `models/Llama-3.2-1B-Instruct/tokenizer.json` exists (skipped in CI). |
| `tests/unit/test_rope.py` | Default and Llama 3 RoPE match Hugging Face cos/sin and rotated q/k within 1e-5. |
| `tests/unit/test_llama.py` | Tiny-model CPU float32 logits within 1e-4 of HF, tied embeddings, cache decode matches a full forward. |
| `tests/unit/test_weights.py` | Safetensors round-trip on the tiny model, tied checkpoint with no `lm_head.weight`, and 1B config fields when the download is present. |
| `tests/unit/test_sampler.py` | Greedy, top-k, top-p, a repeated seed that ignores the global RNG, and a two-row greedy batch. |
| `tests/unit/test_engine.py` | One-at-a-time generation: greedy token ids match Hugging Face exactly on the tiny model, stop ids are excluded, a seed repeats. |
| `tests/unit/test_kv_cache.py` | Fake-memory block counts, the spec's 8B byte-per-token figure, tied-weight dedup, and paged tensor shapes. |
| `tests/unit/test_datasets.py` | Length filtering and fixed-seed subsets for all four workloads, using in-test fixtures. |
| `tests/unit/test_offline.py` | Section 11 JSONL schema on the tiny model (written under `tmp_path`, not committed) and median-repeat selection. |
| `tests/unit/test_baselines.py` | Hugging Face `generate` JSONL shape and a profiler summary, both on the tiny model. |
| `tests/unit/test_vllm_offline.py` | vLLM latency math and fixed-length rows. Does not import vLLM. |
| `tests/unit/test_profile_summary.py` | Idle-gap math, nsys sqlite name lookup, and ncu CSV bandwidth. No GPU and no Nsight binary. |
| `tests/gpu/test_llama_parity.py` | (`gpu`, `slow`) Llama-3.2-1B bf16 greedy parity vs Hugging Face. Skipped when the weights are absent. |
| `tests/gpu/test_engine_generate.py` | (`gpu`, `slow`) `Engine.generate` on one 1B prompt matches a Hugging Face argmax loop. Skipped when the weights are absent. |
| `docs/results/phase1/2026-10-01_p1-4-greedy-parity.json` | Script-written 1B parity result. 10/10 prompts matched on the first 32 of 64 tokens; max absolute logit difference 0.0. |
| `docs/results/phase1/2026-10-02_p1-6-naive-engine.json` | Script-written engine check: one bf16 prompt, 16 greedy tokens, matched Hugging Face. |
| `docs/results/phase2/2026-10-06_p2-3-tinyserve-offline.jsonl` | Script-written naive TinyServe offline run on Llama-3.2-1B bf16. Synthetic, 8 requests, prompt 64, output 32, warmup 2, 3 repeats. Median output throughput is the summary line. Dirty run. |
| `docs/results/phase2/2026-10-06_p2-3-hf-generate.jsonl` | Same requests through Hugging Face `generate`. Median output throughput is the summary line. Dirty run. |
| `docs/results/phase2/2026-10-06_p2-3-profiler.json` | Script-written profiler table for one TinyServe request of that shape. Self CPU time exceeds self CUDA time. `aten::mm` is most of the device time. |
| `docs/results/phase2/2026-10-06_p2-4-vllm-offline.jsonl` | Script-written vLLM offline run, same synthetic ids, one batched `generate` per repeat. Median output throughput is the summary line. ITL percentiles are null. Dirty run. |
| `docs/results/phase2/2026-10-06_p2-5-nsys.json` | Script-written Nsight Systems summary of one naive 1B request (prompt 64, 16 new tokens). Idle fraction is in the summary. Dirty run. No ncu file: counters are blocked on the Windows host. |
| `tests/gpu/test_env_info_gpu.py` | (`gpu`) GPU fields are populated |

## 7. Environment and hardware

- Dev machine: Windows 11 laptop, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, compute capability 8.9), driver 595.97.
- **Development happens in WSL2 Ubuntu 24.04** (D-007): 22 CPU cores and about 15 GB RAM visible, GPU passed through. torch 2.14.0+cu130 with triton 3.8.0; a Triton kernel compiles and runs. Tools in WSL: uv 0.12 (`~/.local/bin`), gh 2.45 (logged in as `devwo1f`), git, gcc 13 (`build-essential`). No CUDA toolkit or `nvcc` is needed for Triton.
- Native Windows (the old `D:\Projects\TinyServe` copy) also runs torch with CUDA, but has no Triton; it is not used for development anymore.
- `HF_TOKEN` is in `~/.bashrc` inside WSL (`chmod 600`, never committed). Llama 3.2 gating group is approved. `models/Llama-3.2-1B-Instruct/model.safetensors` is downloaded (gitignored). Llama 3.1 8B still needs its own access request before the final benchmarks.
- Tools: git 2.52, GitHub CLI 2.87 (logged in as `devwo1f`), uv 0.11. System Python is 3.13; project pins Python 3.11 via uv.
- Implications:
  - Llama-3.2-1B-Instruct in bf16 (~2.5 GB) fits: Phase 1 to 7 development can be local.
  - 3B target + 1B draft (~9 GB bf16) does not fit in 8 GB: needs INT8 target or a cloud GPU (open question).
  - Final Llama-3.1-8B benchmarks need an A100/H100 (cloud).
  - Triton has no official native-Windows support and vLLM needs Linux: hence WSL2 (D-007).
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
- D-008: `transformers==5.17.0` for the tokenizer and the Hugging Face numerical reference.
- D-009: HumanEval is the official JSONL release, not the Hugging Face parquet, so the dataset loader stays dependency-free.
- D-007: develop in WSL2 Ubuntu 24.04 at `~/TinyServe` (resolves Q-001).
- Repo is public on GitHub (human choice). Phase 0 was reviewed by starting Phase 1.

## 11. Open questions / known issues

- Q-002: how to fit 3B target + 1B draft for speculative decoding on an 8 GB GPU (INT8 target vs cloud GPU; decide before Phase 8).
- Llama 3.1 8B access is still needed before final benchmarks (the 3.2 gating group does not cover it).
- The HF token was pasted into a chat. The human should revoke it and put a new one in `~/.bashrc`.
- Nsight Compute still cannot read GPU performance counters until NVIDIA App allows them for all users. `bash scripts/profile_ncu.sh` writes the missing file after that. Phase 3 does not wait on it.

## 12. Next steps

1. P3.2 block manager: allocate, free, reference counts, truncate. Unit tests for exhaustion, exact block boundaries, and truncate.
2. Human: revoke the HF token that was pasted in chat and replace the `HF_TOKEN` line in `~/.bashrc`. Request Llama 3.1 8B access before the final benchmarks. Allow GPU performance counters in NVIDIA App when an ncu summary is wanted.

## 13. Change log

- 2026-09-28 bootstrap: git repo, public GitHub remote, CONTEXT.md, Cursor workflow rule, .gitignore, .gitattributes.
- 2026-09-28 P0.1: package skeleton, pyproject/uv (Python 3.11), ruff + pytest config, tiny_llama.json fixture, smoke tests.
- 2026-09-28 P0.2: spec moved to docs/SPEC.md; PROGRESS, DECISIONS (D-001..D-003, Q-001, Q-002), REFERENCES, learn notes, results/writeup dirs, CLAUDE.md, AGENTS.md.
- 2026-09-28 P0.3: scripts/env_info.py, download_models.sh, download_datasets.sh; torch 2.14.0 (cu130/cpu extras) and numpy.
- 2026-09-28 P0.4: tinyserve/config.py (section dataclasses, validation, dotted overrides, CLI flags) + tests.
- 2026-09-28 P0.5: GitHub Actions CI (CPU torch, ruff, non-GPU tests), CI badge. Phase 0 complete.
- 2026-09-28 env: moved development to WSL2 Ubuntu 24.04 (D-007, resolves Q-001); Triton 3.8.0 verified on the RTX 4060.
- 2026-09-28 P0.3: fix download_models.sh globs; Llama-3.2-1B-Instruct weights downloaded locally (not committed).
- 2026-09-29 P1.1: Tokenizer wrapper, chat template, incremental detokenizer; transformers 5.17.0.
- 2026-09-29 P1.2: RoPE with Llama 3 frequency scaling, matched to Hugging Face within 1e-5.
- 2026-09-29 P1.3: Llama model with contiguous KV cache; CPU float32 logits match HF within 1e-4 on the tiny model.
- 2026-10-01 P1.4: safetensors loader; 1B bf16 greedy parity 10/10 on the first 32 tokens (result file). Single-token decode no longer passes an all-true mask into SDPA.
- 2026-10-01 P1.5: sampler (greedy, temperature, top-k, top-p) and the Section 9 sequence types. Seeds use a per-request generator.
- 2026-10-02 P1.6: naive engine, one request at a time on the contiguous cache. Tiny-model greedy ids match HF; 1B engine check is in the result file. Phase 1 code complete, waiting for the human review gate.
- 2026-10-02 P2.1: workload loader (ShareGPT, code, shared prefix, synthetic) with length filters and a hash-ordered seed. Human asked to start Phase 2. HumanEval download is official JSONL (D-009).
- 2026-10-06 P2.2: offline benchmark writes a Section 11 JSONL (throughput, TTFT, TPOT, ITL). Warmup is a separate pass. No dev-model timing file yet; that is P2.3.
- 2026-10-06 P2.3: naive TinyServe and Hugging Face `generate` baselines on Llama-3.2-1B, plus a profiler table for one request. Numbers are only in the phase2 result files.
- 2026-10-06 P2.4: vLLM 0.31.0 in its own venv, batched offline result on the same synthetic 1B requests. Commands are in `bench/vllm_baseline.md`.
- 2026-10-06 P2.5: Nsight Systems and Nsight Compute wrappers. The nsys summary of one naive 1B request is in the phase2 result file. ncu is blocked on Windows GPU performance counters.
- 2026-10-06 P3.1: paged KV tensors and startup block count from a fake-memory budget. Human asked to start Phase 3.
