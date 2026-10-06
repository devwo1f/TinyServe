# Learning Notes

One note per task or concept, named `<phase>-<task>-<topic>.md` (for example `p3-2-block-manager.md`). These notes are for the human owner: code alone is not enough to learn from.

Each note contains (spec Section 15):

1. **Concept in plain words** (under 200 words), with a small concrete example, e.g. a 3-sequence batch with actual block ids.
2. **Where it lives in the code:** file paths and function names, in reading order.
3. **Key tensors and shapes** as they flow through the task's code.
4. **What was measured** and the result file path (if any).
5. **Pitfalls hit during implementation** and how they were fixed.
6. **Five self-check questions** you should be able to answer in an interview, with short answers inside a collapsed `<details>` block.

## Review gates

Before the next phase starts, you should be able to explain these without notes:

- After Phase 1: the Llama forward pass, GQA, RoPE scaling.
- After Phase 3: block tables, slot mapping, why paging reduces fragmentation, prefix hashing.
- After Phase 4: how the scheduler fills a step, chunked prefill's effect on tail latency, preemption.
- After Phase 5: why decode is memory-bound, how the paged decode kernel iterates, online softmax, how bandwidth utilization is computed.
- After Phase 6: what CUDA graphs remove and why they need static shapes.
- After Phase 7: TTFT vs TPOT vs goodput, how admission control predicts TTFT.
- After Phase 8: why rejection sampling preserves the target distribution, why speculation helps less at large batch sizes.
- After Phase 9: why weight-only quantization speeds up decode but not prefill as much.

Confirm a gate by adding `## <YYYY-MM-DD> | HUMAN | Gate after Phase <n> passed` to `docs/PROGRESS.md`.

## Index

| Note | Topic |
|---|---|
| [p0-setup.md](p0-setup.md) | Phase 0: repo layout, uv, pytest markers, the tiny test model, git workflow |
| [p1-1-tokenizer.md](p1-1-tokenizer.md) | Tokenizer, chat template, streaming detokenization |
| [p1-2-rope.md](p1-2-rope.md) | RoPE and Llama 3 frequency scaling |
| [p1-3-llama.md](p1-3-llama.md) | Llama forward pass, GQA, SwiGLU, contiguous KV cache |
| [p1-4-weights.md](p1-4-weights.md) | Safetensors loading and bf16 greedy parity on Llama 3.2 1B |
| [p1-5-sampler.md](p1-5-sampler.md) | Greedy, temperature, top-k, top-p, per-request seeds |
| [p1-6-naive-engine.md](p1-6-naive-engine.md) | One-request generation on the contiguous cache |
| [p2-1-datasets.md](p2-1-datasets.md) | ShareGPT, code, shared-prefix, and synthetic workloads |
| [p2-2-offline.md](p2-2-offline.md) | Offline throughput and the Section 11 result file |
| [p2-3-baselines.md](p2-3-baselines.md) | Naive TinyServe and Hugging Face generate baselines, and where the time goes |
| [p2-4-vllm.md](p2-4-vllm.md) | vLLM offline baseline and why its token rate is not a kernel-only comparison |
