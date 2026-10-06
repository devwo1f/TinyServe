# P2.2 Offline benchmark

## 1. Concept in plain words

Offline throughput is the engine with nothing else in the way. There is no arrival process and no HTTP client. One request runs to completion, then the next starts. That number is the ceiling: a serving benchmark with a queue cannot go faster than this on the same model and the same lengths.

Each request still has its own latency. TTFT is the wait until the first new token. E2E is the wait until the last new token. TPOT is the average gap after the first token: `(e2e - ttft) / (num_output_tokens - 1)`. ITL is every one of those gaps, not the average, so a stall shows up as a high p99.

Warmup is a separate pass over the first few samples. Those runs are not rows in the file, and the measured pass still includes every sample. Three repeats are written. The summary's percentiles come from the middle repeat after sorting by output tokens per second. The spread is the fastest repeat minus the slowest.

Example: two output tokens. TTFT is 0.10 s, the second token arrives at 0.15 s. E2E is 0.15 s, there is one ITL of 0.05 s, and TPOT is 0.05 s. A single output token has no TPOT.

## 2. Where it lives in the code

1. `Engine._run_one` in `tinyserve/engine/engine.py` records `ttft_s`, `e2e_s`, and `itl_s` on `GenerationResult`.
2. `_sync` waits for the GPU before a clock read. `.item()` already waits for the sampled id.
3. `run_offline` in `bench/offline.py` calls `generate_tokens` once per sample, because each sample has its own `output_len`.
4. `_summary` picks the median repeat and computes p50, p90, and p99.
5. The file is JSONL: one meta line, one request line per measured sample per repeat, one summary line.

## 3. Key tensors and shapes

The benchmark does not build new tensors. Each step still follows the engine: prefill `input_ids` is `[1, prompt_len]`, each decode step is `[1, 1]`, and the logits row that is sampled is `[vocab]`.

## 4. What was measured

No committed result file. The unit test writes a JSONL under pytest's temporary directory and checks that the summary divides the file's own token counts by the file's own wall clock. A tiny random model is not a baseline. The Llama 3.2 1B numbers are the next task (P2.3).

## 5. Pitfalls hit

A GPU forward returns before the kernels finish. Reading the clock there records the launch, not the work. The engine synchronizes before it stamps a token.

The engine still forwards the last token so its keys and values sit in the cache. That token already exists, so that extra forward is not part of E2E. It is part of the run's wall clock, because the process did the work.

`request_rate = inf` means offline. JSON has no infinity, so the writer stores the string `"inf"`.

## 6. Self-check questions

1. Why is offline throughput a ceiling for the serving benchmark?
2. Why is TPOT undefined when a request emits one token?
3. Why are the summary percentiles taken from one repeat instead of pooling all three?
4. Why synchronize before reading the clock on CUDA?
5. Why is the cache write after the last token outside E2E but inside the wall clock?

<details>
<summary>Answers</summary>

1. Serving adds a queue, HTTP, and scheduling. None of that can make the model faster than running requests back to back.
2. TPOT excludes the first token, so the denominator is `num_output_tokens - 1`. One token leaves nothing to average.
3. The spec says report the median run and the spread. Pooling mixes a slow run into the percentiles and hides that spread.
4. The Python call returns when the kernels are queued. An unsynced clock measures the launch.
5. E2E ends when the last token exists. The following forward only stores it for a next token that will not be sampled. The wall clock is how long the run actually took, so that write stays in the throughput denominator.

</details>
