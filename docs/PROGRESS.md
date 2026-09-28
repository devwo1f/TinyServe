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
