# P5.2 In-place RoPE

## 1. Concept in plain words

RoPE rotates pairs of channels in a head by an angle that grows with the token position. The model already builds `inv_freq`, including Llama 3's stretch of the long wavelengths. The kernel only needs that vector and the position of each token.

One program handles one head of one token. It computes cos and sin in float32, casts them to the activation dtype, and writes both halves back into the same memory. Query and key are two launches, because they have different head counts.

A single decode token moves a few kilobytes. The time difference against PyTorch is the launch and the tensors the reference allocates. A long prefill is already a wide PyTorch op, and this kernel is only a little faster there. Attention still calls the PyTorch helper.

## 2. Where it lives in the code

1. `tinyserve/model/rope.py` builds `inv_freq` and is the Hugging Face-checked rotate-half formula.
2. `tinyserve/kernels/reference.py` `apply_rope` applies that formula to `[num_tokens, num_heads, head_dim]` and returns a new tensor.
3. `tinyserve/kernels/rope.py` `apply_rope_` writes the same values into Q and K. CUDA launches `_rope_kernel`. CPU copies the reference into the caller's storage.
4. `bench/microbench/rope.py` times both with `triton.testing.do_bench`.

## 3. Key tensors and shapes

`query` is `[num_tokens, num_heads, head_dim]`. `key` is `[num_tokens, num_kv_heads, head_dim]`. `positions` is `[num_tokens]` int64. `inv_freq` is `[head_dim / 2]` float32. The two halves that get rotated are each `[head_dim / 2]`.

## 4. What was measured

`docs/results/phase5/2026-10-23_p5-2-rope.jsonl`. bf16, RTX 4060 Laptop, dirty at `c2ce895`. Three `do_bench` calls per shape. Bandwidth is algorithm bytes over the median time, divided by 256e9. Algorithm bytes read and write Q and K once and read positions and `inv_freq` once. Nsight Compute did not run.

Llama-3.2-1B shape, 32 query heads, 8 KV heads, head dim 64:

| tokens | fused ms | PyTorch ms | utilization |
|---|---|---|---|
| 1 | 0.015359999611973763 | 0.2908160090446472 | 0.0026387533218688495 |
| 8 | 0.016383999958634377 | 0.2734079957008362 | 0.019577026416614737 |
| 32 | 0.02252800017595291 | 0.3164159953594208 | 0.056884765180706676 |
| 512 | 0.14028799533843994 | 0.3399679958820343 | 0.14610302150624435 |
| 2048 | 0.5038080215454102 | 0.5652480125427246 | 0.16272964401899748 |

At one token, PyTorch divided by fused is 18.93333440047381. At 2048 tokens it is 1.1219511964276587. Achieved bandwidth at 2048 tokens is 41658788868.86336 bytes/s. The highest utilization in the file is 0.2953254758855431, at 512 tokens with head dim 128 (an 8B-shaped head): fused 0.13875199854373932 ms, PyTorch 0.40243199467658997 ms.

## 5. Pitfalls hit

Cos and sin have to be cast to the activation dtype before the multiply. Leaving them in float32 promotes the product and misses the path `Attention` uses.

The two halves of a head must be loaded before either is stored. Writing the first half first would destroy the value the second half still needs.

One program per token, looping over heads, leaves a one-token step with a single program. The committed kernel is one program per head so that step still spreads across the heads.

The model was not switched over. The kernel matches the helper, and the forward still calls the helper.

## 6. Self-check questions

1. What does one program rotate, and what are the two writes?
2. Why is position 0 a no-op?
3. Why is the one-token speedup large and the 2048-token speedup small?
4. Where does Llama 3 scaling enter this kernel?
5. Why is 0.16272964401899748 not an Nsight Compute DRAM fraction?

<details>
<summary>Answers</summary>

1. One head of one token. It writes `x0 * cos - x1 * sin` into the first half and `x1 * cos + x0 * sin` into the second.
2. The angle is `position * inv_freq`. Position 0 makes every angle 0, so cos is 1 and sin is 0, and the head is unchanged.
3. One token moves 10376 algorithm bytes and finishes in 0.015359999611973763 ms. PyTorch spends 0.2908160090446472 ms building new tensors. At 2048 tokens both sides are moving 20988032 bytes, and the times are 0.5038080215454102 ms and 0.5652480125427246 ms.
4. It does not. `compute_inv_freq` folds the wavelength stretch into the vector the kernel multiplies by the position.
5. It is algorithm bytes divided by `do_bench` time, then by 256e9. Counters are blocked, and a head is small, so most of a short step is launch rather than DRAM.

</details>
