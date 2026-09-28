# TinyServe Design Decisions

Every design decision that affects more than one module, every new dependency, every loosened test tolerance, and every contract change (spec Section 9) is recorded here.

Format: `D-<n>` for decisions, `Q-<n>` for open questions. Resolved questions become decisions and link back.

---

## Decisions

### D-001: Python 3.11 managed by uv; project is not an installable package (yet)
- **Date:** 2026-09-28 (P0.1)
- **Context:** Spec Section 5 requires Python 3.11 with uv. The dev machine's system Python is 3.13.
- **Decision:** `requires-python = ">=3.11,<3.12"`, `.python-version` pinned to `3.11`, uv downloads the interpreter. No `[build-system]` yet; tests import the repo via pytest `pythonpath = ["."]`. Revisit if CLI entry points or a wheel are needed.
- **Dependencies added:** dev group only: `pytest` (testing, spec Section 5) and `ruff` (lint/format, spec Section 5). Exact versions are locked in `uv.lock`.

### D-002: Ruff only checks Python files
- **Date:** 2026-09-28 (P0.1)
- **Context:** Recent ruff versions also format Python code blocks inside Markdown, which would rewrite the spec's code snippets.
- **Decision:** `[tool.ruff] include = ["*.py", "*.pyi", "**/pyproject.toml"]`.

### D-003: Git workflow is branch per task, push every commit, merge by PR
- **Date:** 2026-09-28 (bootstrap, human choice)
- **Decision:** Branches `p<phase>-<task>-<slug>`; every commit is pushed immediately; at task end a PR is opened and merged to `main` with a merge commit (`gh pr merge --merge --delete-branch`) once acceptance criteria and CI pass. `CONTEXT.md` is updated in the same commit as every change so any agent can pick up the project. `.gitattributes` forces LF line endings because scripts run on Linux/WSL/CI.

### D-004: PyTorch 2.14.0 via two mutually exclusive extras (`cu130`, `cpu`); numpy as a base dependency
- **Date:** 2026-09-28 (P0.3)
- **Context:** GPU machines need CUDA wheels, while CI needs small CPU wheels. PyPI's default torch is CUDA-only on Linux and CPU-only on Windows, so relying on it gives different builds on different machines.
- **Decision:** `torch==2.14.0` pinned exactly (latest stable at setup time), resolved from the official PyTorch indexes: `uv sync --extra cu130` on GPU machines (CUDA 13.0; the dev machine's driver 595.97 supports it) and `uv sync --extra cpu` in CI. The extras are declared as conflicting in `[tool.uv]`. Plain `uv run` keeps an already installed torch (inexact sync).
- **Dependencies added:** `torch` (spec Section 5); `numpy>=2.0` (torch warns without it; benchmarks and statistics tests need it).
- **Not added:** `huggingface_hub`. The download scripts use `uv run --with huggingface_hub hf download`, so it does not become a project dependency.
- **Triton:** not installed on native Windows (official wheels are Linux-only; see Q-001). On Linux, torch's CUDA wheel pulls in the matching Triton automatically.

### D-005: `scripts/` is an importable package
- **Date:** 2026-09-28 (P0.3)
- **Decision:** `scripts/__init__.py` exists so benchmark code can embed `scripts.env_info.collect_env_info()` in result files (spec Section 11) and tests can import it.

---

## Open questions

### Q-001: WSL2 or native Windows for GPU work? (decide before Phase 1)
- **Context:** Dev machine is Windows 11 with an RTX 4060 Laptop GPU (8 GB). Triton has no official native-Windows wheels (a community `triton-windows` fork exists), `torch.compile`/Triton kernels and CUDA graphs are best supported on Linux, Nsight tools work on both, and vLLM (the baseline, Phase 2.4) requires Linux.
- **Options:**
  1. **WSL2 (Ubuntu) for everything from Phase 1** (recommended): official PyTorch + Triton wheels, vLLM runs, same environment as CI and cloud GPUs. Cost: one-time setup; model files should live in the WSL filesystem for speed.
  2. Native Windows with `triton-windows`: no WSL setup, but an unofficial dependency, possible kernel/compiler differences, and vLLM still needs WSL or a cloud machine.
  3. Native Windows for CPU work, cloud Linux GPU for all GPU work.
- **Status:** open, waiting for the human.

### Q-002: How to run speculative decoding dev setup (3B target + 1B draft) on 8 GB? (decide before Phase 8)
- **Context:** In bf16, Llama-3.2-3B (~6.4 GB) + Llama-3.2-1B (~2.5 GB) weights alone exceed 8 GB before any KV cache.
- **Options:**
  1. 1B target with a smaller draft: not possible within the Llama 3 tokenizer family (1B is the smallest).
  2. INT8 weight-only 3B target (Phase 9 work pulled earlier) + bf16 1B draft: roughly 3.2 GB + 2.5 GB, leaves some room for KV.
  3. Rent a cloud GPU (24 GB+) for Phase 8 development.
- **Status:** open, waiting for the human.

---

## Loosened tolerances

None yet.
