# P3.4 Reference paged attention

## 1. Concept in plain words

Paged attention does not attend over the physical layout. It walks each sequence's block table, copies those tokens back into logical order, and then runs ordinary attention.

The new query tokens are already in the cache. They are the last positions of the context. A decode step has one query token, and it can see every earlier key. A chunk of several new tokens, as in chunked prefill or speculative verification, sits at the end of a longer prefix. Each query token can see keys up to its own position and not past it.

Example, block size 4. Sequence A has context 17 and block table `[3, 1, 6, 0, 4]`. Logical position 16 is physical block 4, offset 0. The other offsets of that block are not part of the context, even if they hold leftover numbers. Sequence B has context 6 and table `[2, 5]`. Both decode with one query token. The outputs match attention over those same tokens laid out contiguously.

## 2. Where it lives in the code

1. `gather_paged_kv` in `tinyserve/kernels/reference.py` copies one sequence's pages and trims the last block to `seq_len`.
2. `standard_attention` is the same three SDPA cases as `Attention.forward` in `tinyserve/model/llama.py`.
3. `paged_attention` gathers K and V for each sequence, then calls `standard_attention`.
4. The model still runs on `ContiguousKVCache`. Wiring this into the forward pass is the next task.

## 3. Key tensors and shapes

`k_cache` and `v_cache` are `[num_blocks, block_size, num_kv_heads, head_dim]`, one layer.

`query` is `[num_seqs, num_heads, q_len, head_dim]`. Every sequence in the call has the same `q_len`.

`gather_paged_kv` returns `[seq_len, num_kv_heads, head_dim]`. Attention wants heads before the sequence, so that becomes `[1, num_kv_heads, seq_len, head_dim]`. GQA repeats KV heads out to `num_heads`. The output is `[num_seqs, num_heads, q_len, head_dim]`.

## 4. What was measured

No result file. The tests are the check: float32 outputs are identical to contiguous attention for a one-token decode, for a query chunk over a prefix, and for a prefill that fills its blocks exactly.

## 5. Pitfalls hit

`is_causal=True` lines the query up with the start of the keys. A short query over a cached prefix has to use a mask that puts the queries at the end. The contiguous model already does that, and this path calls the same helper.

The unused tail of the last block is not context. Gathering a whole page and then attending over it would mix in whatever was left in those slots. The gather stops at `seq_len`.

A block table is not padded with 0. Block 0 is a real block.

## 6. Self-check questions

1. Context 17, block size 4, table `[3, 1, 6, 0, 4]`. Which physical block and offset hold logical position 16?
2. Why does a decode query of length 1 attend to every key, while a chunk of length 3 does not?
3. Why must the unused offsets of the last block be left out of the gather?
4. A query chunk of length 3 ends at position 12. If the key at position 12 changes, which query rows may change?
5. Why does this file call the same SDPA cases as the contiguous `Attention` module?

<details>
<summary>Answers</summary>

1. `16 // 4 = 4`, so the fifth table entry, physical block 4. `16 % 4 = 0`, so offset 0.
2. The single new token is the last position, so every stored key is in its past. A chunk has several positions; an earlier query in that chunk must not see a later key.
3. Those offsets are not part of the sequence. They can hold zeros or a previous sequence's values, and either one would change the softmax.
4. Only the last query row. It sits at position 12. The earlier rows in the chunk are positions 10 and 11, which cannot see position 12.
5. The acceptance test is an exact float32 match. A different mask or a different scale would drift even when the gather is right.

</details>
