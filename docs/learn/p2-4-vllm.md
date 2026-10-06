# P2.4 vLLM baseline

## 1. Concept in plain words

vLLM is the number TinyServe is trying to approach. It is a separate program, with its own torch build, run on the same model and the same token ids.

The comparison is not "the same loop, faster kernels." TinyServe's P2.3 run finished one request before starting the next. vLLM receives all eight requests in one `generate` call, keeps them in a batch, and uses CUDA graphs. A higher token rate from that setup is expected. The useful fact is how large the gap is, and which of those differences caused it.

Example: eight requests of 32 new tokens is 256 output tokens. If they run one after another, the wall clock is about eight prefill-and-decode passes. If they run together, the wall clock is about one prefill of the batch plus 32 decode steps that advance every request.

## 2. Where it lives in the code

1. `bench/vllm_baseline.md` is the install and the exact command.
2. `bench/vllm_offline.py` builds the same synthetic samples as P2.3 and calls `LLM.generate`.
3. `latency_seconds` reads `first_token_latency` (wall clock) and adds `last_token_ts - first_token_ts` (engine-core monotonic clock). Those two clocks are not the same, so they are not subtracted.
4. The JSONL writer is the one in `bench/offline.py`.

## 3. Key tensors and shapes

vLLM owns its own tensors. The prompts we send are eight lists of 64 token ids. Each request is asked for 32 new tokens. `max_model_len` is 512, which is enough for 64 + 32 and is what the KV pool was sized from. TinyServe's naive cache for this workload was only the length of one request.

## 4. What was measured

`docs/results/phase2/2026-10-06_p2-4-vllm-offline.jsonl`

vLLM 0.31.0, torch 2.13.0+cu129, bf16 Llama-3.2-1B-Instruct, RTX 4060 Laptop, driver 595.97. The run was dirty (commit `b3784af` plus this diff).

Median repeat output throughput 517.4217102549717 tokens/s. Total throughput 1552.265130764915 tokens/s. Spread 42.912821521890066. Wall clock 0.4947608400000263 s for 256 output tokens. TTFT p50 0.05819261074066162 s. E2E p50 0.49397697574067934 s. TPOT p50 0.014057560161290894 s. ITL percentiles are null.

The P2.3 TinyServe file, same token ids, one request at a time, median output throughput 47.23866681302909 tokens/s.

## 5. Pitfalls hit

The first `import vllm` failed because torchcodec 0.17 wants `libnvrtc.so.13` and this venv's torch is CUDA 12.9. Pinning torchcodec to 0.14.0 fixed the import.

The first engine start then failed inside FlashInfer's sampler, which JIT-compiles with `nvcc`. There is no CUDA toolkit on this machine. `VLLM_USE_FLASHINFER_SAMPLER=0` uses the other sampler. Temperature 0 is still greedy.

The offline `LLM` object disables request stats unless `disable_log_stats=False`. With the default, `metrics` is `None` and there is no TTFT to record.

## 6. Self-check questions

1. Why is vLLM in a different virtualenv?
2. Why is 517 tokens/s not "vLLM's kernels are 11 times faster"?
3. Why are the ITL percentiles null?
4. Why set `max_model_len` to 512?
5. Why add a monotonic gap to a wall-clock TTFT instead of subtracting the two token timestamps from arrival time?

<details>
<summary>Answers</summary>

1. vLLM pins its own torch (here 2.13.0+cu129). Installing it into TinyServe's environment would replace torch 2.14.0+cu130.
2. The TinyServe number is one request at a time. The vLLM number is eight requests in one batched `generate`, with CUDA graphs. The token rate includes that scheduling difference.
3. The request stats store the first token and the last token. They do not store every gap. TPOT is still the mean gap, `(e2e - ttft) / (num_output_tokens - 1)`.
4. These prompts need 96 positions. vLLM reserves KV for `max_model_len`. 131072 would ask for a much larger cache than this comparison uses. 512 is recorded in the result config.
5. `arrival_time` is wall clock. `first_token_ts` and `last_token_ts` are a different monotonic clock. Subtracting across clocks is meaningless. The gap between the two monotonic stamps is a duration, and adding it to the wall-clock TTFT keeps end-to-end latency on the arrival timeline.

</details>
