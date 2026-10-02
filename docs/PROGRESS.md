# TinyServe Progress Log

Session log, newest entries first. Template (spec Appendix B):

```markdown
## <YYYY-MM-DD> | <agent: Claude Code / Codex / Cursor> | Task P<x>.<y>

**Status:** in progress / done / blocked
**What changed:** files touched and a two-line summary
**Tests:** which tests were added, and pass/fail status (CPU and GPU)
**Results:** result file paths, if any (no numbers without a file)
**Decisions:** links to DECISIONS.md entries, if any
**Next step:** the exact next action for whoever continues
**Questions for the human:** anything needing a decision
```

Review gate confirmation (written by the human only):

```markdown
## <YYYY-MM-DD> | HUMAN | Gate after Phase <n> passed
```

---

## 2026-10-01 | Cursor | Task P1.5

**Status:** done
**What changed:** `tinyserve/engine/sequence.py` holds `SamplingParams`, `SequenceStatus`, and `Sequence` (spec Section 9). `tinyserve/engine/sampler.py` draws one token: temperature 0 is argmax; otherwise temperature, top-k, then top-p, using a per-request generator.
**Tests:** `tests/unit/test_sampler.py`. CPU: defaults, greedy, top-k, top-p, repeated seed independent of the global RNG, two-row greedy batch.
**Results:** none
**Decisions:** none
**Next step:** P1.6 naive engine: `generate` one request at a time with the contiguous cache, greedy parity with Hugging Face.
**Questions for the human:** none

## 2026-10-01 | Cursor | Task P1.4

**Status:** done
**What changed:** `tinyserve/model/weights.py` loads safetensors onto the target device and dtype and re-ties `lm_head` when the checkpoint omits it. Single-token decode in `Attention.forward` no longer passes an all-true mask into SDPA; that mask was forcing the math kernel and bf16 tokens drifted off Hugging Face.
**Tests:** `tests/unit/test_weights.py` (CPU round-trip, tied checkpoint, 1B config shape). `tests/gpu/test_llama_parity.py`: 1 passed. CPU suite still green.
**Results:** `docs/results/phase1/2026-10-01_p1-4-greedy-parity.json` — 10/10 prompts matched on the first 32 of 64 greedy tokens; max absolute logit difference 0.0. The run was dirty (measured `9d09770` plus this diff).
**Decisions:** none
**Next step:** P1.5 sampler: greedy, temperature, top-k, top-p, per-request seeds.
**Questions for the human:** none

## 2026-09-29 | Cursor | Task P1.3

**Status:** done
**What changed:** `tinyserve/model/llama.py`: RMSNorm, GQA attention, SwiGLU, contiguous KV cache, LM head with optional tied embeddings. Parameter names match Hugging Face.
**Tests:** `tests/unit/test_llama.py`. CPU float32 on the tiny model: prefill logits within 1e-4 of HF (eager attention), tied-embedding variant the same, and prefill-plus-decode matches a single full forward.
**Results:** none
**Decisions:** none
**Next step:** P1.4 load Llama-3.2-1B safetensors and run the GPU greedy parity test from spec Section 12.
**Questions for the human:** none

## 2026-09-29 | Cursor | Task P1.2

**Status:** done
**What changed:** `tinyserve/model/rope.py`: inverse frequencies, Llama 3 wavelength scaling, cos/sin caches, and `apply_rotary`.
**Tests:** `tests/unit/test_rope.py`. CPU float32: default RoPE and Llama 3 scaling match Hugging Face cos/sin and rotated q/k within 1e-5, including positions 8192, 8193, and 20000 (past the original context of 8192).
**Results:** none
**Decisions:** none. Conceptual note in REFERENCES.md.
**Next step:** P1.3 Llama model with contiguous KV cache, CPU float32 logit parity under 1e-4 on the tiny model.
**Questions for the human:** none

## 2026-09-29 | HUMAN (via session) | Gate after Phase 0 passed

Human asked to start Phase 1.

## 2026-09-29 | Cursor | Task P1.1

**Status:** done
**What changed:** `tinyserve/model/tokenizer.py`: `Tokenizer` (encode, decode, chat template, eos/bos) and `IncrementalDetokenizer` (holds back a trailing U+FFFD so a character split across tokens is not streamed as a replacement box). Added `transformers==5.17.0` (D-008).
**Tests:** `tests/unit/test_tokenizer.py`. CPU: byte-level fixture tests always run; Llama round-trip, chat template, and incremental non-ASCII run when `models/Llama-3.2-1B-Instruct` is present (passed in WSL). CI skips the Llama tests because the tokenizer is gated and not committed.
**Results:** none
**Decisions:** D-008
**Next step:** P1.2 RoPE, including Llama 3 frequency scaling, matched against Hugging Face in float32.
**Questions for the human:** none

## 2026-09-28 | Cursor | P0.3 follow-up: model download

**Status:** done
**What changed:** `scripts/download_models.sh` repeats `--include` once per glob (the CLI was treating extra patterns as filenames and skipping the weights) and no longer passes the token on the command line. Downloaded `models/Llama-3.2-1B-Instruct` including `model.safetensors` (gitignored).
**Tests:** no code tests; download completed and the safetensors file is present. Ruff/pytest unchanged (shell script only).
**Results:** none
**Decisions:** none
**Next step:** human reviews Phase 0, then P1.1 tokenizer wrapper.
**Questions for the human:** revoke the HF token that was pasted in chat and replace it in `~/.bashrc`. Llama 3.1 8B access is still separate.

## 2026-09-28 | Cursor | Environment: WSL2 (P0.3 follow-up)

**Status:** done
**What changed:** installed WSL2 Ubuntu 24.04, `build-essential`, `git`, `gh`, and uv; cloned the repo to `~/TinyServe`; `uv sync --extra cu130`. Recorded D-007 (resolves Q-001) and updated CONTEXT.md.
**Tests:** in WSL: `uv run pytest -m "not gpu"` 17 passed, `-m gpu` 1 passed, ruff clean. `env_info.py` reports triton 3.8.0. A throwaway Triton vector-add kernel compiled and matched torch on the GPU (not committed).
**Results:** none
**Decisions:** D-007
**Next step:** once the human has Llama access and `HF_TOKEN`, download the 1B model and start P1.1.
**Questions for the human:** Phase 0 review; Q-002 (before Phase 8).

## 2026-09-28 | Cursor | Task P0.5

**Status:** done (Phase 0 complete; waiting for human review before Phase 1)
**What changed:** `.github/workflows/ci.yml` (ubuntu-latest, `uv sync --locked --extra cpu`, env_info, `ruff check`, `ruff format --check`, `pytest -m "not gpu"`); CI badge in README.
**Tests:** CI on PR #5: pass (torch 2.14.0+cpu on Linux, 17 passed, 1 GPU test deselected). CI on `main` is checked after merge.
**Results:** none
**Decisions:** none new
**Next step:** human reviews Phase 0 and decides Q-001 (WSL2 vs native Windows). Then P1.1 tokenizer wrapper (needs `HF_TOKEN` and the Llama license accepted).
**Questions for the human:** Q-001 (before Phase 1), Q-002 (before Phase 8).

## 2026-09-28 | Cursor | Task P0.4

**Status:** done
**What changed:** `tinyserve/config.py`: `TinyServeConfig` with Model/Cache/Scheduler/Speculative/Server/Benchmark dataclasses, validation, `apply_overrides`, `add_config_args`, `config_from_args`, `to_dict`.
**Tests:** `tests/unit/test_config.py` (defaults, JSON serialization, typed string parsing, immutability of base, unknown field, 6 invalid-value cases, CLI flags). CPU: 17 passed.
**Results:** none
**Decisions:** D-006
**Next step:** P0.5 GitHub Actions CI.
**Questions for the human:** none

## 2026-09-28 | Cursor | Task P0.3

**Status:** done
**What changed:** `scripts/env_info.py` (+ `scripts/__init__.py`), `scripts/download_models.sh` (dev / dev-spec / final presets, requires `HF_TOKEN`), `scripts/download_datasets.sh` (ShareGPT, WikiText-2, HumanEval); `pyproject.toml` gains `torch==2.14.0` via `cpu`/`cu130` extras and `numpy`.
**Tests:** `tests/unit/test_env_info.py` (all fields present, `--json` CLI), `tests/gpu/test_env_info_gpu.py` (GPU fields populated). CPU: 4 passed. GPU (RTX 4060 Laptop, native Windows): 1 passed. `env_info.py` prints every field on the GPU machine (`triton` is None on native Windows, as expected). Manually checked: `download_datasets.sh humaneval` downloads, `download_models.sh` exits with an error when `HF_TOKEN` is unset.
**Results:** none
**Decisions:** D-004, D-005
**Next step:** P0.4 `tinyserve/config.py`.
**Questions for the human:** Q-001 still open (WSL2 recommended for Triton).

## 2026-09-28 | Cursor | Task P0.2

**Status:** done
**What changed:** moved the spec to `docs/SPEC.md`; created `docs/PROGRESS.md`, `docs/DECISIONS.md`, `docs/REFERENCES.md`, `docs/learn/README.md`, `docs/learn/p0-setup.md`, `docs/results/`, `docs/writeup/`, `CLAUDE.md`, `AGENTS.md`. CLAUDE.md/AGENTS.md follow Appendix A plus a pointer to `CONTEXT.md`.
**Tests:** no new tests (docs only); `uv run pytest -m "not gpu"` 2 passed; ruff clean.
**Results:** none
**Decisions:** D-001, D-002, D-003; open questions Q-001, Q-002
**Next step:** P0.3 environment scripts (`scripts/env_info.py`, download scripts, torch dependency).
**Questions for the human:** Q-001 (WSL2 vs native Windows) and Q-002 (fitting 3B + 1B in 8 GB) before Phase 1 / Phase 8.

## 2026-09-28 | Cursor | Task P0.1

**Status:** done
**What changed:** package skeleton (`tinyserve/` subpackages, `bench/`, `eval/`, `tests/`), `pyproject.toml` with uv (Python 3.11), ruff config, pytest markers `gpu`/`slow`, `tests/fixtures/tiny_llama.json`. Merged in PR #1.
**Tests:** `tests/unit/test_skeleton.py` (imports, fixture shape); 2 passed on CPU; no GPU tests yet.
**Results:** none
**Decisions:** D-001, D-002
**Next step:** P0.2 docs scaffolding.
**Questions for the human:** none

## 2026-09-28 | Cursor | Bootstrap

**Status:** done
**What changed:** `git init`, public GitHub repo https://github.com/devwo1f/TinyServe, `CONTEXT.md`, `.cursor/rules/tinyserve-workflow.mdc`, `.gitignore`, `.gitattributes`.
**Tests:** none
**Results:** none
**Decisions:** D-003
**Next step:** P0.1 project skeleton.
**Questions for the human:** none
