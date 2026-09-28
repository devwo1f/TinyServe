# Phase 0: Repository and Environment Setup

## 1. Concept in plain words

Phase 0 has no ML in it. It builds the scaffolding that keeps the later phases honest and reproducible:

- **uv** manages the Python 3.11 interpreter and a locked dependency set (`uv.lock`), so every machine (your laptop, WSL, CI, a rented GPU) runs the same versions.
- **pytest markers** split tests into CPU tests (always run) and `gpu`/`slow` tests. Most engine logic (block manager, scheduler, sampler, rejection sampling) is pure bookkeeping and can be tested on CPU with a tiny random model.
- **The tiny model** (`tests/fixtures/tiny_llama.json`) is a Llama config small enough to run instantly: 2 layers, hidden 64, 4 query heads sharing 2 KV heads (GQA group size 2), head_dim 16, vocab 256. Example: with block_size 16, a 20-token prompt needs 2 KV blocks per layer, each block holding `16 x 2 heads x 16 dims` values for K and again for V.
- **Docs as memory:** `CONTEXT.md` (current state), `docs/PROGRESS.md` (session log), `docs/DECISIONS.md` (why), `docs/learn/` (what you should understand).

## 2. Where it lives in the code

1. `pyproject.toml`: dependencies, pytest markers, ruff config.
2. `tests/fixtures/tiny_llama.json`: the tiny model config.
3. `tests/unit/test_skeleton.py`: smoke tests.
4. `CONTEXT.md`, `.cursor/rules/tinyserve-workflow.mdc`, `CLAUDE.md`, `AGENTS.md`: agent and workflow instructions.
5. `scripts/env_info.py` (P0.3) and `tinyserve/config.py` (P0.4): added later in this phase.

## 3. Key tensors and shapes

None in Phase 0. The first tensors appear in Phase 1 (RoPE cos/sin caches `[max_position, head_dim]`).

## 4. What was measured

Nothing. No benchmark numbers exist yet.

## 5. Pitfalls hit

- Recent ruff versions also reformat Python code blocks inside Markdown, which would rewrite the spec's snippets. Fixed by restricting ruff to Python files (DECISIONS.md D-002).
- On Windows, uv warns that it cannot hardlink from its cache when the project is on a different drive. It is harmless; set `UV_LINK_MODE=copy` to silence it.

## 6. Self-check questions

1. Why do CPU unit tests use a tiny random model instead of the real 1B model?
2. What does `--strict-markers` protect against?
3. Why is `uv.lock` committed?
4. With the tiny config, how many query heads share each KV head, and why does that matter for the KV cache size?
5. Why must benchmark numbers only ever come from committed result files?

<details>
<summary>Answers</summary>

1. The bookkeeping logic doesn't depend on model size. A tiny model runs in milliseconds on any machine including CI, and float32 on CPU is deterministic, so exact-match tests are possible.
2. A typo such as `@pytest.mark.gppu` becomes an error instead of silently creating a new marker, which would make a GPU test run (and fail) in the CPU suite.
3. It pins every transitive dependency version, so results and test behavior are reproducible across machines and over time.
4. 4 query heads / 2 KV heads = 2 query heads per KV head. The KV cache stores only KV heads, so GQA halves the KV memory compared with full multi-head attention here (and shrinks it 4x for Llama-3.1-8B: 32 query heads, 8 KV heads).
5. It makes every claim auditable and reproducible, and it prevents accidental (or convenient) fabrication. It is one of the project's honesty constraints.

</details>
