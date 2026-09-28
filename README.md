# TinyServe

A small, readable LLM inference serving engine built from scratch in Python, PyTorch, and Triton. It serves a Llama-architecture model on a single GPU through an OpenAI-compatible streaming API.

Status: early development (Phase 0, repository and environment). See [CONTEXT.md](CONTEXT.md) for the current state of the project and [docs/SPEC.md](docs/SPEC.md) for the full specification.

No benchmark numbers are reported here until they come from committed result files under `docs/results/`.

## Development

```bash
uv sync
uv run ruff check .
uv run pytest -m "not gpu"
```
