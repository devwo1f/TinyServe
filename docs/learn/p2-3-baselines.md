# P2.3 Baselines

## 1. Concept in plain words

A baseline is the number you are not allowed to forget. Later kernels and schedulers are only faster if they beat this run on the same model, the same lengths, and the same machine.

Two engines see the same synthetic requests: 8 prompts of 64 tokens, 32 new tokens each, greedy, end-of-sequence ignored. Warmup is a separate pass of 2 requests. Three repeats are written. The summary uses the middle repeat by output tokens per second.

One extra request is passed through `torch.profiler`. That file is where the time went, not a second throughput number.

Example: a decode step multiplies each weight matrix by a single token. That is a matrix-vector product, not a big square multiply. There are thousands of them in one short request, and each one is a launch.

## 2. Where it lives in the code

1. `run_offline` in `bench/offline.py` times TinyServe.
2. `run_hf_generate` in `bench/baselines.py` calls Hugging Face `generate` with `do_sample=False` and `min_new_tokens` equal to `max_new_tokens`.
3. `_TokenClock` drops the first streamer call, which is the prompt, then stamps each new token.
4. `profile_generation` writes the profiler table for one TinyServe request.
5. `_group_self_time` buckets `aten::` ops only. Kernel rows repeat time the aten op already counted.

## 3. Key tensors and shapes

Each request is batch 1. Prefill `input_ids` is `[1, 64]`. Each decode step is `[1, 1]`. The sampled row is `[vocab]`. The profiler counted 3729 `aten::mm` calls on the profiled request: one prefill plus 32 decode forwards, and each forward has the layer projections plus the LM head.

## 4. What was measured

Script: `python -m bench.baselines --model models/Llama-3.2-1B-Instruct`. bf16 Llama-3.2-1B-Instruct. The run was dirty (commit `76ea9ec` plus this diff).

`docs/results/phase2/2026-10-06_p2-3-tinyserve-offline.jsonl`

Median repeat output throughput 47.23866681302909 tokens/s. Total throughput 141.71600043908728 tokens/s. Spread 1.8621044186043605. Wall clock 5.419289266000021 s for 256 output tokens. TTFT p50 0.020378449999981285 s. E2E p50 0.655839892500012 s. TPOT p50 0.02046503843548415 s.

`docs/results/phase2/2026-10-06_p2-3-hf-generate.jsonl`

Median repeat output throughput 36.88392486381207 tokens/s. Total throughput 110.65177459143622 tokens/s. Spread 10.029779902509528. Wall clock 6.940693023999984 s. TTFT p50 0.03122180749997483 s. E2E p50 0.8499828670000227 s. TPOT p50 0.026557511161290716 s.

`docs/results/phase2/2026-10-06_p2-3-profiler.json`

One request, prompt length 64, 32 new tokens. The table footer is self CPU time 865.858 ms and self CUDA time 368.889 ms. `aten::mm` is 336.677 ms, 91.27% of that CUDA total, 3729 calls. The two largest kernels under it are cublas gemv: 243.407 ms (65.98%) and 82.022 ms (22.24%). Those percentages overlap `aten::mm`; they are the same matmuls, not extra work. The aten-only groups in the file put 0.9126788294415407 of device self time in gemm and 0.01466236255574756 in attention.

## 5. Pitfalls hit

The host time is larger than the device time. A naive loop launches a small gemv per projection per token, and the CPU spends longer dispatching that than the GPU spends inside the kernel.

The first grouper added those gemv kernel rows on top of `aten::mm`, so the fractions overlapped. The committed file only buckets `aten::` names.

RMSNorm does not appear as an op named "norm". It is mean, multiply, and rsqrt, which land in the elementwise and other buckets. A zero norm bucket does not mean the norm is free.

Hugging Face `generate` is not a bare forward. The shipped generation config samples, so the baseline forces greedy. It also copies logits to float32 and synchronizes to hand each token to the streamer. TinyServe synchronizes too, because the latency clock would otherwise record the launch. The two throughputs are still the numbers above.

## 6. Self-check questions

1. Why is this offline run the number later optimizations have to beat?
2. Why is decode a gemv here?
3. Why can CPU time exceed GPU time when the model is running on the GPU?
4. Why set `min_new_tokens` equal to `max_new_tokens` for the Hugging Face baseline?
5. Why do the gemv percentages in the profiler table overlap `aten::mm`?

<details>
<summary>Answers</summary>

1. Same model, dtype, lengths, and machine. A later change is faster only if its result file beats this one under those conditions.
2. Batch size is 1 and each decode step has one token, so each linear layer multiplies a weight matrix by a vector.
3. Each tiny kernel has a launch. Thousands of launches make the host the longer part of the timeline. The profile's self CPU total is 865.858 ms and self CUDA total is 368.889 ms.
4. That stops `generate` from ending on an end-of-sequence id, so both engines emit the dataset length. `do_sample=False` is also required, because the checkpoint's generation config samples.
5. The kernel row is the GPU work that `aten::mm` already counts as its own device time. Adding them is counting one matmul twice.

</details>
