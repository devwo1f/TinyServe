# P2.5 Profiling scripts

## 1. Concept in plain words

A timeline says when the GPU was empty. `torch.profiler` adds up kernel time. It does not show the holes between launches. Those holes are the CPU deciding the next tiny matmul and waiting for the previous one.

Example: two kernels, one from 0 to 10 ns and the next from 20 to 30 ns. The GPU worked for 20 ns and sat idle for 10 ns. The span is 30 ns. Overlapping kernels count once, so a second kernel that starts at 5 ns and ends at 12 ns does not add a second copy of the overlap.

Nsight Systems records every kernel's start and end. Nsight Compute is the other tool: it replays a few launches and reads hardware counters (bytes moved, occupancy). It is not a throughput number.

## 2. Where it lives in the code

1. `scripts/profile_target.py` loads the naive engine, warms up one request, then calls `torch.cuda.profiler.start` around a second request of the same shape.
2. `scripts/profile_nsys.sh` runs Nsight Systems on that command with `--capture-range=cudaProfilerApi`, exports sqlite, and calls `summarize_nsys_sqlite`.
3. `scripts/profile_ncu.sh` runs Nsight Compute on the first few launches inside that same range.
4. `coverage_ns` in `scripts/profile_summary.py` turns start/end stamps into span, busy, and idle.
5. `scripts/profile_tool.sh` finds `nsys`, `ncu`, or the project python. The tools live under `$HOME/opt`, not in the repo.

## 3. Key tensors and shapes

The captured request is batch 1. The nsys run used prompt ids of shape `[1, 64]` and then 16 decode steps of `[1, 1]`. The sampled row each step is `[vocab]`. Nsight Compute was aimed at a one-token prompt so the sampled launches would be matrix-vector products. That run did not collect counters.

## 4. What was measured

`docs/results/phase2/2026-10-06_p2-5-nsys.json`

Nsight Systems 2026.3.2.476, torch 2.14.0+cu130, bf16 Llama-3.2-1B-Instruct, RTX 4060 Laptop, driver 595.97. One greedy request, prompt 64, 16 new tokens, after one uncaptured warmup of the same shape. The run was dirty (commit `a6cc7bb` plus this diff).

11252 kernel launches. Span 0.478305183 s. GPU busy 0.212564769 s. GPU idle inside that span 0.265740414 s. Idle fraction 0.5555875692862814.

The largest kernel is cublas `internal::gemvx::kernel` (the first `top_kernels` row): 1040 calls, 0.137198408 s. The next gemvx row is 768 calls, 0.04702683 s. A bf16 tensor-core gemm is 64 calls, 0.00617003 s. Flash-attention is 256 calls, 0.002596841 s.

There is no Nsight Compute result file. `ncu` printed `ERR_NVGPUCTRPERM` and the script exited before writing JSON. The same error happened as root inside WSL. The Windows driver is what allows the counters.

## 5. Pitfalls hit

Ubuntu's apt package of Nsight is the 2022 build. This machine's torch is CUDA 13, so the wrappers look for the 2026 redistributable unpacked under `$HOME/opt`.

On WSL, Nsight's default GPU clock conversion is wrong. The nsys wrapper sets `CuptiUseRawGpuTimestamps=false`, which is the setting Nsight documents for WSL.

`ncu` as root in WSL still cannot read performance counters. NVIDIA App (driver 595): System, Advanced, Developer, Manage GPU Performance Counters, allow all users. After that, `bash scripts/profile_ncu.sh` is the command that writes the missing file.

## 6. Self-check questions

1. Why does an idle gap matter if the kernel itself is fast?
2. Why is the idle time the span minus the union of kernel intervals, not the sum of the kernel durations subtracted twice?
3. Why is the Nsight capture wrapped in `cudaProfilerStart` / `Stop`?
4. Why does Nsight Compute not replace the offline throughput run?
5. Why did `sudo` inside WSL not fix `ERR_NVGPUCTRPERM`?

<details>
<summary>Answers</summary>

1. Decode launches one small gemv per weight matrix per token. If the CPU takes longer to launch than the GPU takes to run, the GPU is idle most of the span. This file's idle fraction is 0.5555875692862814. CUDA graphs (Phase 6) exist to remove that launch gap.
2. Overlapping kernels would be counted twice if you summed durations and subtracted from the span. The union counts each nanosecond once. Idle is then the holes between the first start and the last end.
3. Weight loading is a big copy and is not the serving loop. The profiler range starts after the warmup request, so the timeline is one generation.
4. Compute replays a handful of launches to read counters. It does not time a whole request set, and it cannot run until the Windows driver allows performance counters.
5. WSL uses the Windows GPU driver. The restriction is on that driver. A root shell in Ubuntu is not a Windows administrator to the driver. The switch is in NVIDIA App on the Windows host.

</details>
