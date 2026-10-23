# P5.1 Fused add and RMSNorm

## 1. Concept in plain words

A pre-norm layer adds the sublayer output onto the residual, then the next RMSNorm reads that sum. Two kernels mean a round trip through memory: store the sum, load it again, store the normalized row.

One kernel can add in float32, keep that sum for the variance, and write both the stored residual and the normalized row. The variance is `mean(sum^2)`. The output is `weight * sum * rsqrt(variance + eps)`. The stored residual is the sum rounded back to the model dtype, which is what the next add will read.

A short row does not fill the memory pipe. The launch costs more than the bytes. A long row is where the byte count starts to matter.

## 2. Where it lives in the code

1. `tinyserve/kernels/reference.py` `add_rmsnorm` is the obvious PyTorch version.
2. `tinyserve/kernels/rmsnorm.py` `add_rmsnorm` calls that on CPU and the Triton kernel on CUDA.
3. `_add_rmsnorm_kernel` is one program per row. The first loop writes the sum and accumulates the square. The second loop scales.
4. `algorithm_bytes` and `issued_bytes` are the two byte counts.
5. `bench/microbench/rmsnorm.py` times both paths with `triton.testing.do_bench` and writes the JSONL.

The decoder still calls `RMSNorm`. This kernel is not in the forward yet.

## 3. Key tensors and shapes

`x` and `residual` are `[rows, hidden]`. `weight` is `[hidden]`. The kernel returns the normalized row `[rows, hidden]` and the stored sum `[rows, hidden]`. The variance inside the reference is `[rows, 1]`.

## 4. What was measured

`docs/results/phase5/2026-10-23_p5-1-rmsnorm.jsonl`. bf16, RTX 4060 Laptop, dirty at `8d3b743`. Three `do_bench` calls per shape (warmup 25, rep 100). The time used for bandwidth is the median of those three. Algorithm bytes are `x`, `residual`, and `weight` read once, plus the sum and the norm written. Peak is 256e9 bytes/s: NVIDIA lists this GPU as 8 GB GDDR6 on a 128-bit bus, and 16 Gbps times 128/8 is 256 GB/s. Nsight Compute did not run.

| rows | hidden | fused ms | PyTorch ms | utilization |
|---|---|---|---|---|
| 1 | 2048 | 0.021503999829292297 | 0.13311999291181564 | 0.003720238124770894 |
| 8 | 2048 | 0.020479999482631683 | 0.14284799993038177 | 0.025781250651289174 |
| 32 | 2048 | 0.021503999829292297 | 0.1382399946451187 | 0.09598214361908906 |
| 512 | 2048 | 0.06348799914121628 | 0.26419198513031006 | 0.5163810553720332 |
| 2048 | 2048 | 0.18995200097560883 | 0.9861119985580444 | 0.6901111824393608 |
| 1 | 4096 | 0.02969600073993206 | 0.11724800243973732 | 0.00538793090023226 |
| 32 | 4096 | 0.030207999981939793 | 0.12902399897575378 | 0.13665254245458067 |

At 2048 by 2048 the fused median is 0.18995200097560883 ms and the PyTorch median is 0.9861119985580444 ms. PyTorch divided by fused is 5.191374628818299. Achieved bandwidth there is 176668462704.47635 bytes/s, utilization 0.6901111824393608. One row of 2048 is 0.021503999829292297 ms and utilization 0.003720238124770894: the same launch, almost no bytes. The kernel reloads the row, so DRAM traffic can be higher than `algorithm_bytes`. The utilization above is that lower byte count over wall time, not a counter.

## 5. Pitfalls hit

Triton is imported only when the tensor is on CUDA. A top-level import would break the CPU CI environment.

The second loop reloads `x` and `residual` and adds them again in float32. Reloading the stored sum would normalize the rounded value and miss the reference.

`algorithm_bytes` does not count that reload. `issued_bytes` does. The result file uses `algorithm_bytes`, and the config line says so.

The model forward was left on the PyTorch module. Swapping it in changes the GPU bf16 add (float32 sum versus a bf16 add) and needs a parity run of its own.

## 6. Self-check questions

1. What does one row of the fused kernel compute?
2. Why is the variance taken from the float32 sum, not from the stored residual?
3. Why is utilization near zero for one row and 0.6901111824393608 for 2048 rows?
4. What is the peak in the denominator, and where does 256e9 come from?
5. Why is this number not an Nsight Compute DRAM measurement?

<details>
<summary>Answers</summary>

1. `residual_out = dtype(x + residual)` and `normed = dtype(weight * sum * rsqrt(mean(sum^2) + eps))`, with the sum in float32.
2. The stored residual is already rounded. The reference normalizes the float32 sum. Using the rounded value would not match it.
3. One row moves 20480 bytes. The launch dominates, so bytes per second stay near 952380959.9413489. 2048 rows move 33558528 bytes in 0.18995200097560883 ms, which is 0.6901111824393608 of 256e9.
4. 256e9 bytes/s. NVIDIA lists an 8 GB GDDR6, 128-bit RTX 4060 Laptop GPU. 16 Gbps times 128/8 is 256 GB/s.
5. Counters are blocked on the Windows host. The file divides algorithm bytes by `do_bench` time. The kernel also reloads the row, so real DRAM can be higher.

</details>
