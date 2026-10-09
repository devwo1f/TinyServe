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

## 2026-10-09 | Cursor | Task P4.2

**Status:** done
**What changed:** `Scheduler.schedule` preempts when the free list cannot hold the next token. The victim is the newest running request that this step has not already reserved blocks for. Its blocks are freed, `num_computed_tokens` goes back to 0, and it returns to the waiting queue with status `PREEMPTED`. Tokens it already generated stay on the sequence, and the next prefill runs the prompt and those tokens again. A request that is the only one running is not preempted. `ScheduledBatch.preempted` lists who lost KV this step.
**Tests:** `tests/unit/test_scheduler.py`. CPU: two sequences fill a 2-block pool; the older one needs another block and the newer one is preempted, with its output token kept and its block table cleared. A pool of 2 blocks of size 4 still finishes two requests (prompt 4, 4 new tokens each) with the same token ids as a pool of 8 blocks. The lone-request cases still do not preempt. 12 passed. CPU suite: 120 passed, 3 deselected.
**Results:** none
**Decisions:** none. Conceptual note in REFERENCES.md.
**Next step:** P4.3 engine step loop. `step` schedules, runs, samples, updates, and frees finished sequences. Batched greedy outputs must match one-at-a-time outputs.
**Questions for the human:** none

## 2026-10-07 | HUMAN (via session) | Gate after Phase 3 passed

Human asked to continue into Phase 4.

## 2026-10-07 | Cursor | Task P4.1

**Status:** done
**What changed:** `tinyserve/engine/scheduler.py` keeps a waiting queue and a running queue. `schedule` reserves blocks for decodes first, one token each, then spends the leftover `max_num_batched_tokens` on prefill. With chunking on, a prompt is split to the budget and to the free blocks. With chunking off, the whole remaining prompt has to fit or the step takes nothing from that queue. `max_num_seqs` caps how many sequences are in flight. A running chunk continues before a new request is admitted. Decodes that ran move to the back of the running queue so a tight budget still reaches the others. `num_computed_tokens` is not advanced here. `preempted` is empty.
**Tests:** `tests/unit/test_scheduler.py`. CPU: a decode and a prefill under a budget of 3 are ordered decode-then-chunk and the batch token count is 3; a 10-token prompt with budget 4 runs as 4, 4, 2, then one decode; chunking off leaves a 10-token prompt and the short one behind it waiting; two short prefills share a step; the sequence cap admits one of two; a running chunk is continued before the next wait; three decodes with budget 2 rotate so the third runs on the next step; one free block of 4 caps the chunk at 4 and the next step is empty with no preemption; a full block does not grow a decode and does not preempt; four tokens already computed are not prefilled again. 10 passed. CPU suite: 118 passed, 3 deselected.
**Results:** none
**Decisions:** none. Conceptual note in REFERENCES.md.
**Next step:** P4.2 preemption. When a step cannot allocate, preempt the most recently arrived running sequence, free its blocks, and requeue it. A test that forces block exhaustion still finishes every request.
**Questions for the human:** none

## 2026-10-07 | Cursor | Task P3.7

**Status:** done. Phase 3 code is complete. Do not start Phase 4 until the human writes the review-gate line.
**What changed:** `tinyserve/kv/waste.py` computes the Section 10 fraction. Paged slots are the live block tables times the block size. Contiguous slots charge `max_len` for every live sequence, which is the reservation a dense cache makes at the serving limit (`model.max_model_len`, 4096). The engine records one sample after each step when `record_kv_waste` is set, before the blocks are freed. `bench/kv_waste.py` runs eight variable-length requests on the tiny model and also asks the block manager to hold every finished length at once.
**Tests:** `tests/unit/test_kv_waste.py`. CPU: 15/16/17 tokens allocate 16/16/32 slots; a short block table is rejected; two live sequences of 6 and 4 tokens with block size 4 use 10 slots, hold 12 paged slots, and are charged 32 contiguous slots; engine steps for a 5-token prompt plus 2 output tokens match the block-manager walk (used 5, 6, 7, paged 8). Recording stays empty unless asked. CPU suite: 108 passed, 3 deselected.
**Results:** `docs/results/phase3/2026-10-07_p3-7-kv-waste.jsonl`. 80 steps. One-at-a-time paged waste p50 0.05610119047619048, p90 0.37812500000000027, p99 0.8881249999999996. Contiguous waste p50 0.9754638671875, p90 0.99560546875, p99 0.99956298828125. Concurrent at full length: used 516, paged allocated 576, contiguous allocated 32768, paged waste 0.10416666666666667, contiguous waste 0.9842529296875. The run was dirty (measured `39ddfe3` plus this diff). Slot counts do not depend on the weights; the tiny model is there so the rows come from the engine step loop.
**Decisions:** none. Conceptual note in REFERENCES.md.
**Next step:** Human review gate for Phase 3 (block tables, slot mapping, why paging reduces unused KV, prefix hashing). Then P4.1 scheduler. Do not start Phase 4 before the gate line is in this file.
**Questions for the human:** confirm the Phase 3 gate when you can explain block tables, slot mapping, the waste difference, and prefix hashing without notes.

## 2026-10-07 | Cursor | Task P3.6

**Status:** done
**What changed:** `tinyserve/kv/prefix_cache.py` hashes each full prompt block with its parent hash and maps that to a physical block. `match` stops at the first miss and never returns the block that holds the last prompt token. After a request finishes, full blocks are cached and parked; the rest return to the free list. `Engine.generate_tokens` keeps one pool for the call, shares a hit, and prefills only the tail. If the free list is short, the oldest unused cached block is reclaimed. `BlockManager.share` accepts a parked block. `bench/prefix_cache.py` runs the shared-prefix prompts with caching off, then on.
**Tests:** `tests/unit/test_prefix_cache.py`. CPU: a change in an earlier block misses the later block; an exact multiple of the block size does not reuse the last block; LRU drops the oldest parked block and a parked block can be shared; a second request on the tiny model reuses 8 tokens and its greedy ids match a cold prefill. Existing engine and block-manager tests still pass. CPU suite: 102 passed, 3 deselected. No new GPU parity test; the 1B measurement is the result file.
**Results:** `docs/results/phase3/2026-10-07_p3-6-prefix-cache.jsonl`. Four Llama-3.2-1B prompts, 100 tokens each, 8 new tokens, block size 16. Computed prompt tokens: 400 with caching off, 112 with caching on (288 cached). TTFT seconds, off: p50 0.042021384000008766, p90 0.26767375410003075, p99 0.3537614999100412. On: p50 0.03972630149999645, p90 0.05923054309997156, p99 0.06563750620996131. Greedy ids matched. The run was dirty (measured `51546e1` plus this diff). The off run is first, so its first request includes the cold start.
**Decisions:** none. Conceptual note in REFERENCES.md.
**Next step:** P3.7 memory waste: allocated slots versus used slots, compared with a contiguous max-length allocation. Commit a result file with both numbers.
**Questions for the human:** none

## 2026-10-06 | Cursor | Task P3.5

**Status:** done
**What changed:** `tinyserve/engine/model_runner.py` packs a step into flattened ids, positions, slots, and attention metadata. Block tables are padded with -1. `LlamaForCausalLM.forward_paged` writes each layer into the paged cache and returns logits only at `logits_indices`. `Engine.generate` uses that path, one request at a time, on a pool sized for the request. The contiguous forward remains for the dense reference.
**Tests:** `tests/unit/test_model_runner.py`. CPU: slots and the -1 pad match the spec formula; a 7-token paged prefill matches the contiguous logits exactly in float32; a 3-token prefix plus a 4-token chunk, and a mixed prefill/decode batch, match within 1e-5. Existing engine, llama, and KV tests still pass. GPU: `tests/gpu/test_engine_generate.py` matched Hugging Face greedy tokens on Llama-3.2-1B. The committed phase1 result file was not changed.
**Results:** none
**Decisions:** none
**Next step:** P3.6 prefix cache: hash full blocks, reuse them, and evict with LRU. On the shared-prefix workload, computed prompt tokens should drop; measure the TTFT change and commit the result file.
**Questions for the human:** none

## 2026-10-06 | Cursor | Task P3.4

**Status:** done
**What changed:** `tinyserve/kernels/reference.py` gathers K and V through each sequence's block table, trims the unused tail of the last block, and runs the same three SDPA cases as the contiguous `Attention` module. Query length 1 is a decode step. A longer query is the end of a cached prefix. The naive engine still uses the contiguous cache.
**Tests:** `tests/unit/test_paged_attention.py`. CPU float32: a one-token decode of contexts 17 and 6 on scrambled block tables matches contiguous attention exactly; a query of length 3 over a 13-token context matches, and changing the last key does not change the first query row; an 8-token prefill that fills two blocks matches. A short block table is rejected. 4 passed.
**Results:** none
**Decisions:** none
**Next step:** P3.5 switch the model to paged KV. The model runner builds flattened inputs and attention metadata (spec Section 9). Existing parity tests still have to pass.
**Questions for the human:** none

## 2026-10-06 | Cursor | Task P3.3

**Status:** done
**What changed:** `tinyserve/kernels/kv_store.py` writes new K and V into the paged pool. `slot_mapping` turns a logical position into `block_table[p // block_size] * block_size + (p % block_size)`. `write_kv` scatters a flattened batch with those slots. The naive engine still uses the contiguous cache.
**Tests:** `tests/unit/test_kv_store.py`. CPU: positions 0, 3, and 4 on table `[3, 6]` with block size 4 are slots 12, 15, and 24; a second sequence's position 0 on block 1 is slot 4; those values land on that block and offset and unwritten slots stay zero. A position past the table, a negative position, an out-of-range slot, and a layer mismatch are rejected. 6 passed.
**Results:** none
**Decisions:** none
**Next step:** P3.4 reference paged attention: gather K/V through block tables and match contiguous-cache attention in float32, for query length 1 and for a longer query over a cached prefix.
**Questions for the human:** none

## 2026-10-06 | Cursor | Task P3.2

**Status:** done
**What changed:** `tinyserve/kv/block_manager.py` loans block ids from a free list. `allocate` appends enough ids for `num_computed_tokens + num_new_tokens` and does not mark those tokens computed. `free` and `truncate` decrement ref counts; a block returns to the free list only at zero. `share` is how a second sequence holds the same block. A block the caller marks cached is parked instead, and `reclaim` is the way back. The naive engine still uses the contiguous cache.
**Tests:** `tests/unit/test_block_manager.py`. CPU: 15/16/17 tokens take 1/1/2 blocks, a full block plus one token appends a block, a partial block absorbs one more token, exhaustion allocates nothing, a second `free` is a no-op, a shared block survives `free` and `truncate`, truncate to 0/15/16/17 keeps the right prefix, and a cached block is parked until `reclaim`. 10 passed.
**Results:** none
**Decisions:** none
**Next step:** P3.3 KV store: write new K/V into the paged cache through `slot_mapping`. Test that values land in the correct block and offset.
**Questions for the human:** none

## 2026-10-06 | Cursor | Task P3.1

**Status:** done
**What changed:** `tinyserve/kv/cache.py` allocates paged K and V and computes how many blocks fit. The budget is total GPU memory times utilization, minus weight bytes, the profiled activation peak, and `memory_safety_margin_gib`. Tied embeddings are counted once. `num_gpu_blocks_override` skips the budget. `str(CacheProfile)` is the startup line (block count and token capacity). The naive engine still uses the contiguous cache.
**Tests:** `tests/unit/test_kv_cache.py`. CPU: the spec's 8B token is 131072 bytes (8192 tokens per GiB), a fake budget floors to 90 blocks / 1440 tokens, a short budget is 0 blocks, an override ignores the budget, tied weights are not double-counted, layer views match the spec shape, and a CPU build with a fake activation size allocates that many blocks. 9 passed.
**Results:** none
**Decisions:** none
**Next step:** P3.2 block manager: allocate, free, reference counts, and truncate, with tests for exhaustion, exact block boundaries, and truncate.
**Questions for the human:** none

## 2026-10-06 | HUMAN (via session) | Gate after Phase 2 passed

Human asked to start Phase 3.

## 2026-10-06 | Cursor | Task P2.5

**Status:** done
**What changed:** `scripts/profile_nsys.sh` and `scripts/profile_ncu.sh` profile one naive-engine request. `scripts/profile_target.py` keeps weight loading and one warmup request outside the CUDA profiler range. `scripts/profile_summary.py` writes a small JSON from the nsys sqlite (kernel span, busy union, idle gaps) or from an ncu CSV. Nsight 2026 builds live under `$HOME/opt`, not in the project venv. Raw reports stay in `profiles/`.
**Tests:** `tests/unit/test_profile_summary.py`. CPU: idle-gap math, sqlite kernel names, ncu CSV bandwidth, and `bash -n` on the wrappers. GPU: the nsys wrapper wrote the result file below. `bash scripts/profile_ncu.sh` exited 1 with `ERR_NVGPUCTRPERM`, including when rerun as root in WSL, so there is no ncu result file.
**Results:** `docs/results/phase2/2026-10-06_p2-5-nsys.json`. The run was dirty (measured `a6cc7bb` plus this diff).
**Decisions:** none
**Next step:** Human enables GPU performance counters for all users in NVIDIA App (System > Advanced > Developer > Manage GPU Performance Counters), then reruns `bash scripts/profile_ncu.sh`. Phase 3 does not start until the human confirms the Phase 2 review gate.
**Questions for the human:** Enable the performance-counter switch, then say so and the ncu summary can be recorded. Confirm the Phase 2 gate when you are ready for paged KV cache.

## 2026-10-06 | Cursor | Task P2.4

**Status:** done
**What changed:** vLLM 0.31.0+cu129 is installed in `/home/abhay/venvs/tinyserve-vllm` (not the project venv). `bench/vllm_offline.py` runs the P2.3 synthetic set as one batched `generate` per repeat and writes a Section 11 JSONL. `bench/vllm_baseline.md` records the install, the versions, and the command. FlashInfer's sampler is turned off because it JIT-compiles with `nvcc`, which this machine does not have. torchcodec is pinned to 0.14.0 so the import does not ask for `libnvrtc.so.13`.
**Tests:** `tests/unit/test_vllm_offline.py`. CPU: latency math and fixed output lengths, without importing vLLM. GPU: the script wrote the result file below.
**Results:** `docs/results/phase2/2026-10-06_p2-4-vllm-offline.jsonl`. The run was dirty (measured `b3784af` plus this diff). ITL percentiles are null; vLLM's request stats do not include each gap.
**Decisions:** none
**Next step:** P2.5 profiling scripts: `scripts/profile_nsys.sh` and `scripts/profile_ncu.sh`, and a learning note from a run on the naive engine.
**Questions for the human:** none

## 2026-10-06 | Cursor | Task P2.3

**Status:** done
**What changed:** `bench/baselines.py` runs the same synthetic requests through naive TinyServe and Hugging Face `generate`, one model at a time, and writes Section 11 JSONL. A third file is a `torch.profiler` table for one TinyServe request. `generate` is greedy with `min_new_tokens` equal to `max_new_tokens`, so both sides emit the dataset length.
**Tests:** `tests/unit/test_baselines.py`. CPU: tiny-model `generate` JSONL matches the shared row shape; the profiler summary is JSON and has the aten buckets. GPU: the 1B script wrote the three result files below.
**Results:** `docs/results/phase2/2026-10-06_p2-3-tinyserve-offline.jsonl`, `docs/results/phase2/2026-10-06_p2-3-hf-generate.jsonl`, `docs/results/phase2/2026-10-06_p2-3-profiler.json`. The run was dirty (measured `76ea9ec` plus this diff).
**Decisions:** none
**Next step:** P2.4 vLLM baseline in a separate venv, with `bench/vllm_baseline.md` and an offline result on the dev model.
**Questions for the human:** none

## 2026-10-06 | Cursor | Task P2.2

**Status:** done
**What changed:** `bench/offline.py` runs a fixed sample list through `Engine.generate_tokens` and writes a Section 11 JSONL (config, env info, git commit, one row per request per repeat, summary). `GenerationResult` now carries TTFT, E2E, and inter-token latencies. CUDA work is synchronized before the clock is read. The cache write after the last token is outside those per-request times.
**Tests:** `tests/unit/test_offline.py`. CPU: tiny-model JSONL matches its own token counts and clock; the median repeat is the middle throughput; warmup runs are not rows; an EOS id can stop a request. `tests/unit/test_engine.py` also checks the new timing fields.
**Results:** none. A tiny-model timing file is not a baseline. The dev-model numbers are P2.3.
**Decisions:** none
**Next step:** P2.3 baselines: naive TinyServe and Hugging Face `generate` on the dev model, with a `torch.profiler` note on where time goes.
**Questions for the human:** none

## 2026-10-02 | HUMAN (via session) | Gate after Phase 1 passed

Human asked to start Phase 2.

## 2026-10-02 | Cursor | Task P2.1

**Status:** done
**What changed:** `bench/datasets.py` turns ShareGPT, code JSON/JSONL, a shared token prefix, and synthetic ids into `Sample`s. Prompts and answers over 1024 tokens are dropped (the caps are arguments). A seed picks a hash-ordered subset. `scripts/download_datasets.sh` fetches official HumanEval JSONL (D-009).
**Tests:** `tests/unit/test_datasets.py`. CPU: length filter, first-turn-only ShareGPT, stable subset, JSON and JSONL code rows, shared prefix ids, synthetic lengths and seed. 8 passed.
**Results:** none
**Decisions:** D-009
**Next step:** P2.2 offline benchmark through `Engine.generate_tokens`, writing a Section 11 JSONL result.
**Questions for the human:** none

## 2026-10-02 | Cursor | Task P1.6

**Status:** done. Phase 1 code is complete. Do not start Phase 2 until the human writes the review-gate line.
**What changed:** `tinyserve/engine/engine.py`: `Engine.generate` encodes each prompt and runs it to completion on its own contiguous cache before the next prompt. Stop ids are not emitted. Temperature 0 goes through the sampler's argmax.
**Tests:** `tests/unit/test_engine.py`. CPU: two prompts match a Hugging Face argmax loop exactly, stop id is excluded, a seed repeats, string `generate` matches id `generate_tokens`. `tests/gpu/test_engine_generate.py`: 1 passed.
**Results:** `docs/results/phase1/2026-10-02_p1-6-naive-engine.json` — one bf16 prompt, 16 greedy tokens, matched Hugging Face. The run was dirty (measured `6af5ab9` plus this diff).
**Decisions:** none
**Next step:** Human review gate for Phase 1 (forward pass, GQA, RoPE scaling). Then P2.1 datasets. Do not start Phase 2 before the gate line is in this file.
**Questions for the human:** confirm the Phase 1 gate when you can explain the forward pass, grouped-query attention, and Llama 3 RoPE scaling without notes.

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
