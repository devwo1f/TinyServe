# P3.3 KV store

## 1. Concept in plain words

The block manager hands out physical block ids. The KV store is what actually writes the vectors into those blocks.

A sequence does not own a contiguous slice of the pool. Logical position `p` lives in `block_table[p // block_size]` at offset `p % block_size`. The flat slot is `block_id * block_size + offset`. Every new token in the batch, from every sequence, becomes one entry in that slot list, and the write is one index assignment per layer.

Example, block size 4.

- Sequence A holds physical blocks `[3, 6]` and writes positions 0, 3, and 4. Those slots are 12, 15, and 24: block 3 offset 0, block 3 offset 3, block 6 offset 0.
- Sequence B holds block `[1]` and writes position 0. That slot is 4.
- The batch slot list is `[12, 15, 24, 4]`. Token 0 of the flattened batch lands in block 3, not in block 0.

The other offsets of block 3 stay zero. An unused block stays zero.

## 2. Where it lives in the code

1. `slot_mapping` in `tinyserve/kernels/kv_store.py` turns one sequence's positions and block table into slots.
2. `write_kv` scatters `key` and `value` into `PagedKVCache` with those slots.
3. The cache tensors themselves are still `PagedKVCache` in `tinyserve/kv/cache.py`. This task does not allocate them and does not call the block manager.

## 3. Key tensors and shapes

`key` and `value` are `[num_layers, num_tokens, num_kv_heads, head_dim]`. That is the batch after sequences are concatenated, not attention's `[batch, num_kv_heads, seq, head_dim]`.

`slots` is `[num_tokens]` int64.

For each layer the pool is viewed as `[num_blocks * block_size, num_kv_heads, head_dim]`, and `pool[slots]` receives that layer's tokens. The same slots are used for every layer and for both K and V.

## 4. What was measured

No result file. The placement test is the check: the four slots above land on blocks 3, 3, 6, and 1 at the offsets named, and unwritten slots stay zero.

## 5. Pitfalls hit

The block index is into the block table, not into the pool. Position 4 with table `[3, 6]` reads `block_table[1]`, which is physical block 6. Using 1 as the block id would write the wrong page.

`write_kv` does not bump `num_computed_tokens`. The forward stores K and V first; the caller marks those tokens computed only after the step succeeds.

A later Triton kernel has to match this indexing path. This file stays the reference.

## 6. Self-check questions

1. With block size 16, where does logical position 17 go if the block table is `[4, 9]`?
2. Why is the slot `block_id * block_size + offset` instead of the logical position?
3. Why can two sequences in one batch write with one `slots` tensor?
4. Why does an unwritten offset of a used block stay zero?
5. Why does this write not set `num_computed_tokens`?

<details>
<summary>Answers</summary>

1. Block index `17 // 16 = 1`, offset `17 % 16 = 1`, so physical block 9, slot `9 * 16 + 1`.
2. Logical position 17 is not pool index 17. The physical block id can be any free block the manager handed out.
3. The batch is already flattened. Each token carries its own slot, so the sequences do not need separate writes.
4. The cache is allocated as zeros, and the write only touches the listed slots.
5. The slots are filled during the forward. If the step fails after the write, the manager still treats those tokens as not computed until the caller says otherwise.

</details>
