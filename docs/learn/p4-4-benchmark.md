# P4.4 Throughput and latency

## 1. Concept in plain words

A step can run many requests together. The serial engine finishes one request, then starts the next, so the GPU sees one sequence while the others wait. One shared step loop keeps every admitted request moving. More tokens leave per second, and each request also waits behind the others, so the gaps between its tokens grow.

Arrivals can be all at once, or spread out. A Poisson process draws the gaps. At a low rate the GPU sits idle between arrivals, so throughput falls and the gaps stay small. At a high rate the requests pile up, throughput matches the offline batch, and the tail gap grows.

Chunk size is the token budget of one step. Eight prompts of 128 tokens do not fit in a budget of 32, so prefill is split across steps. A budget of 512 finishes those prompts in fewer steps. On this run the tail inter-token gap barely moved. Time to first token and throughput did.

## 2. Where it lives in the code

1. `Engine.prepare_pool` sizes the KV pool for every prompt before the first arrival.
2. `Engine.add_request` queues one prompt when it arrives.
3. `Engine.has_pending` is false only when both queues are empty. That is the only time the arrival loop sleeps.
4. `Engine.step` is one iteration. `Engine.results` reads finished sequences in caller order.
5. `bench/offline.py` `run_offline` is still one request per call. That is the naive series.
6. `bench/batching.py` `poisson_offsets` rebuilds one arrival schedule from a seed. `run_concurrent` times one step loop per repeat.
7. `bench/vllm_offline.py` writes the vLLM file after TinyServe has exited.
8. `bench/plots.py` reads the JSONL files and writes the png files. It does not measure.

## 3. Key tensors and shapes

The forward is the same flattened batch as P4.3. `input_ids` and `slot_mapping` are `[num_tokens]` for the step. Eight decodes are 8 tokens. Eight prompts of 64 tokens under a budget of 2048 are one prefill step of 512 tokens. The chunk sweep uses prompts of 128, so eight prompts are 1024 prompt tokens. A budget of 32 runs at most 32 of those tokens in a step.

## 4. What was measured

Llama-3.2-1B-Instruct, bf16, RTX 4060 Laptop. Synthetic, 8 requests, temperature 0, ignore EOS, prefix caching off, chunked prefill on, warmup 2, 3 repeats. The median repeat by output throughput is the summary line. The run was dirty (`9f0ba5a` plus this diff). vLLM 0.31.0 ran in its own process after TinyServe exited. Same request ids as the Phase 2 synthetic set (seed 0, prompt 64, output 32), except the chunk sweep (seed 1, prompt 128, output 32).

Plots, drawn by `python -m bench.plots` from these files:

- `docs/results/phase4/p4-4-offline-throughput.png`
- `docs/results/phase4/p4-4-poisson.png`
- `docs/results/phase4/p4-4-chunk-itl.png`

vLLM does not record each inter-token gap, so its ITL percentiles are null and the offline figure is throughput only.

Offline output throughput, tokens/s:

| File | output throughput | p99 ITL (s) | TTFT p50 (s) |
|---|---|---|---|
| `2026-10-10_p4-4-naive-offline.jsonl` | 27.151815607747235 | 0.055324154120002904 | 0.03655654900001082 |
| `2026-10-10_p4-4-batched-offline.jsonl` | 79.23819762893967 | 0.12375314453998698 | 0.2175509059999996 |
| `2026-10-10_p4-4-vllm-offline.jsonl` | 538.9213492983187 | null | 0.057195305824279785 |

Batched TinyServe divided by this vLLM file is 0.14703109782551507. Batched divided by the serial remeasure is 2.9183388239543957. The Phase 2 serial file is a different engine: `docs/results/phase2/2026-10-06_p2-3-tinyserve-offline.jsonl` records 47.23866681302909. The Phase 2 vLLM file was not edited; it records 517.4217102549717. This comparison uses the new files.

Poisson, same 64/32 requests. Output throughput includes the idle time between arrivals:

| rate (1/s) | output throughput | p99 ITL (s) |
|---|---|---|
| 1 | 43.997764526519234 | 0.07549776608000684 |
| 4 | 71.05169340225018 | 0.1219061062900073 |
| 16 | 79.0944460802214 | 0.14259723525999476 |

Rate 16 is already next to the offline batched throughput. Past that, more arrivals wait; they do not add tokens per second.

Chunk sweep, prompts of 128, 32 new tokens, all at once:

| max batched tokens | output throughput | p99 ITL (s) | TTFT p99 (s) |
|---|---|---|---|
| 32 | 51.53778178624679 | 0.13736097405001488 | 2.949519835640038 |
| 128 | 69.29115908890815 | 0.12643900614000245 | 0.692552819600009 |
| 512 | 79.43782375918802 | 0.127288989740008 | 0.3120244181799671 |

Smaller chunks did not lower p99 ITL here. Budget 32 is slightly worse (0.13736097405001488 versus 0.127288989740008). What moved is the first token and the token rate: TTFT p99 goes from 2.949519835640038 seconds to 0.3120244181799671, and output throughput goes from 51.53778178624679 to 79.43782375918802. A short budget still spends the step on prefill, so a decode that has started waits almost as long between tokens. It just waits through more steps before the first token exists.

The comparison-set config records `greedy_ids_match_serial` false. That check is outside the timed pass. CPU float32 tests on the tiny model still match one-at-a-time greedy ids. The 1B bf16 batch did not. The bool is in the JSONL. The throughput rows still count the tokens the engine emitted.

## 5. Pitfalls hit

Sleeping while a request is in flight adds that pause to its inter-token gaps. The loop calls `step` whenever `has_pending` is true, and sleeps only when both queues are empty.

A later Poisson arrival must not grow the pool. `prepare_pool` sizes it for every prompt before the clock starts, so pages already in use stay put.

A 64-token prompt never chunks under a budget of 2048. The chunk files use 128-token prompts and budgets of 32, 128, and 512.

`run_offline` stays one request per call. The batched offline number is `run_concurrent` with no arrival offsets. Mixing those two would hide the concurrency gain.

vLLM was a second process. The two models were not loaded together.

Plots ran after the vLLM file existed, so the bar chart includes it. The png files are not hand-typed series.

## 6. Self-check questions

1. Why can output throughput rise while p99 inter-token latency also rises?
2. Why is a Poisson rate of 1 slower, in tokens per second, than the offline batch?
3. What did chunk size change on this run, and what did it leave almost alone?
4. Why is 47.23866681302909 not the serial number in this comparison?
5. Why does the offline plot omit vLLM's p99 ITL?

<details>
<summary>Answers</summary>

1. The GPU does more requests per step, so more tokens finish per second. Each request waits behind the others in that step, so its own gaps get longer.
2. The wall clock includes the idle gaps between arrivals. Tokens per second fall even though each request is less crowded. Rate 16 has already caught the offline batch, which is saturation.
3. Budget 32 versus 512 moved TTFT p99 from 2.949519835640038 s to 0.3120244181799671 s and output throughput from 51.53778178624679 to 79.43782375918802. p99 ITL stayed near 0.13 s and was slightly worse at the small budget.
4. That number is the Phase 2 contiguous engine, one request at a time. This serial file is the current paged engine, still one request at a time, and it records 27.151815607747235.
5. vLLM's request stats expose the first token and the last token. The ITL list is empty, so the percentiles are null. The chart is throughput only.

</details>
