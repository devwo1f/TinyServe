# P3.1 Paged KV cache sizing

## 1. Concept in plain words

The contiguous cache is one long strip per sequence, reserved out to a maximum length. A 6-token request that was allowed 512 positions still holds 512 slots. Paging cuts the strip into blocks of 16 tokens and only hands out as many blocks as the sequence has used.

The pool is sized once, at startup, from whatever GPU memory is left after the weights and a single profiled forward. That leftover is turned into a whole number of blocks. The startup line is how many blocks that is, and how many tokens those blocks can hold.

Example: block size 16, and a sequence whose logical positions are 0 through 17. Position 17 is block index `17 // 16 = 1`, offset `17 % 16 = 1`. If the block manager later assigns physical block 4 to that logical block, the slot is `4 * 16 + 1 = 65`. This task only allocates the pool. It does not assign blocks yet.

A fake budget shows the arithmetic. Total memory 1000000 bytes, utilization 0.9, weights 100000, activations 50000, safety margin 10000. Leftover is 740000 bytes. At 8192 bytes per block that is 90 blocks, 1440 tokens. The remainder 6400 bytes is not a block, so it is not used.

## 2. Where it lives in the code

1. `bytes_per_token` and `kv_budget_bytes` in `tinyserve/kv/cache.py`.
2. `num_blocks_for_budget` floors the budget to whole blocks.
3. `weight_bytes` skips a parameter it has already counted, which is how a tied LM head stays one matrix.
4. `cache_profile` packages the figures. `str` of that object is the startup line.
5. `PagedKVCache` holds `k` and `v`. `layer_kv` is one layer's `[num_blocks, block_size, num_kv_heads, head_dim]`.
6. `build_paged_cache` is what startup calls. Pass `activation_bytes` on CPU. On CUDA, leaving it empty runs one prefill with no KV cache.

## 3. Key tensors and shapes

For the tiny model, 3 blocks, block size 16, 2 layers, 2 KV heads, head dim 16:

- `k` and `v`: `[2, 3, 16, 2, 16]` which is `[num_layers, num_blocks, block_size, num_kv_heads, head_dim]`
- one layer: `[3, 16, 2, 16]`
- capacity: `3 * 16 = 48` tokens in the whole pool

The 8B reference shape is the same layout with 32 layers, 8 KV heads, head dim 128, bf16. One token is 131072 bytes. 1 GiB holds 8192 tokens.

## 4. What was measured

No GPU result file. The unit test checks the 8B byte count and the fake budget above (90 blocks, 1440 tokens). A tiny-model build on CPU with a made-up activation size allocates the block count that formula returns.

## 5. Pitfalls hit

Summing `parameters()` counts a tied embedding twice, because the LM head is the same tensor yielded again. The KV budget would then be short by a whole vocabulary matrix. `weight_bytes` keeps one copy.

Profiling a forward that also allocates a contiguous KV cache would treat that cache as an activation and then refuse to allocate a similar cache for real. The profile prefill passes `cache=None`, so the peak is activations.

On CPU there is no CUDA peak stat. Returning 0 would give the cache the entire budget and the first real batch would run out of memory. CPU callers pass an activation number, and the CUDA measurer refuses to run on CPU.

## 6. Self-check questions

1. Why is a short sequence cheaper in a paged cache than in the contiguous cache?
2. Why subtract activations and a safety margin, not only the weights?
3. Why does position 17 not sit at the start of a block when the block size is 16?
4. Why is a tied LM head counted once?
5. Why does the profile forward pass `cache=None`?

<details>
<summary>Answers</summary>

1. The contiguous cache reserves the maximum length. Paging reserves 16-token blocks only for positions the sequence has reached, and those blocks come from one shared pool.
2. The forward needs workspace besides the weights. One profile pass can also underestimate a real batch, so the config holds back `memory_safety_margin_gib`. The leftover, floored to whole blocks, is the KV pool.
3. `17 // 16` is block index 1 and `17 % 16` is offset 1. The first block holds positions 0 through 15.
4. The LM head weight is the embedding matrix. `parameters()` lists that storage twice, but the GPU only holds it once.
5. A temporary contiguous cache inside the profile would be counted as activation memory. The real paged cache is allocated after the profile, from what is left.

</details>
