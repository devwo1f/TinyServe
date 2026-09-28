# TinyServe agent instructions

The full specification is in docs/SPEC.md. It is the source of truth.
The current end-to-end state of the project is in CONTEXT.md. Read it first, and update it in the same commit as every change.

At the start of every session:
1. Read CONTEXT.md.
2. Read docs/SPEC.md Section 3 (Agent Rules).
3. Read the latest entries in docs/PROGRESS.md and all of docs/DECISIONS.md.
4. Pick the next unclaimed task from docs/SPEC.md Section 13 and claim it in PROGRESS.md.

Hard rules:
- Never fabricate or hand-edit benchmark numbers. Numbers come only from committed result files.
- Never copy code from vLLM, SGLang, Nano-vLLM, or other projects. Record conceptual borrowing in docs/REFERENCES.md.
- Every optimized path needs a reference implementation and a comparison test.
- Shape comments on tensor code, docstrings explaining why.
- Run `uv run pytest -m "not gpu"` before every commit.
- End every session with a PROGRESS.md entry and, for finished tasks, a docs/learn/ note.
- Do not start a new phase until the human has confirmed the review gate in PROGRESS.md.

Git workflow:
- Branch per task: `p<phase>-<task>-<slug>`. Commit messages: `P<phase>.<task>: <short description>`.
- Push after every commit. At task end, open a PR to main and merge once CI is green.
