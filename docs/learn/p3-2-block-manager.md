# P3.2 Block manager

## 1. Concept in plain words

The KV pool is a stack of numbered blocks. The block manager is the librarian: it loans block ids, counts how many sequences still need each one, and takes a block back only when the count hits zero.

A block holds 16 tokens. A sequence of 6 tokens needs two blocks: positions 0–3 in the first, positions 4–5 in the second. The last block is only partly full. Asking for one more token does not take a third block until position 16 is passed.

Example, block size 4, five blocks numbered 0 through 4.

- Sequence A asks for 6 tokens. It receives `[0, 1]`.
- Sequence B asks for 4 tokens. It receives `[2]`.
- Sequence C asks for 8 tokens. It receives `[3, 4]`. The free list is empty.
- C asks for one more token. That would be a ninth token, which needs a third block. `can_allocate` is false, and the free list is left as it was.

If D reuses A's block 0, that block's count is 2. Freeing A drops it to 1 and returns block 1 (count 1, and only A held it). Block 0 stays with D.

## 2. Where it lives in the code

1. `blocks_for_tokens` in `tinyserve/kv/block_manager.py` is the ceiling division.
2. `BlockManager.can_allocate` and `allocate` append only the extra ids. They do not set `num_computed_tokens`.
3. `share` bumps the count when a second sequence reuses a block.
4. `free` drops every id on the sequence. A count that is still positive stays off the free list.
5. `truncate` drops blocks that sit entirely past the new length, and clips `num_computed_tokens` so it cannot point past that length.
6. A block the caller marks cached is parked instead of freed. `reclaim` is how that block comes back. Prefix caching is the caller, and it is not built yet.

## 3. Key tensors and shapes

This task does not write the cache tensor. It writes `seq.block_table`, a list of integer block ids.

With block size 16, a table `[0, 1]` covers 32 token slots. Logical position 17 is block index `17 // 16 = 1`, offset `17 % 16 = 1`, so physical block `block_table[1]`.

## 4. What was measured

No result file. The tests are the check: 15, 16, and 17 tokens take 1, 1, and 2 blocks; a request that needs one more block than the pool fails without taking any; a shared block survives `free` and `truncate` until the last holder lets go.

## 5. Pitfalls hit

`allocate` must not mark the tokens computed. The slots are reserved before the forward runs. If the manager set `num_computed_tokens` early, a failed step would look finished.

Truncating to an exact multiple keeps the last full block. Length 16 with block size 16 is one block, not zero. Length 0 is the case that drops every block.

A cached block at count 0 is not free. Putting it on the free list would let a new sequence overwrite a prefix someone might still match. `reclaim` is the only way back, and it refuses a block that is still referenced or already free.

## 6. Self-check questions

1. Why does a 16-token sequence with block size 16 hold one block, while 17 tokens hold two?
2. Why does `can_allocate` return true for one more token when the last block still has an empty slot and the free list is empty?
3. Why does freeing one of two sequences that share a block not return that block?
4. What does truncate to 16 do to a sequence that held 32 tokens?
5. Why is a cached block with ref count 0 not on the free list?

<details>
<summary>Answers</summary>

1. Sixteen tokens fill one block exactly. The seventeenth token starts the next block. `blocks_for_tokens` is ceiling division, and zero tokens hold nothing.
2. The new token fits in the block the sequence already holds, so the extra-block count is 0. An empty free list only matters when a new id is required.
3. The ref count is how many sequences still name that id. It returns to the pool at 0, not when the first sequence leaves.
4. Blocks past the new length are dropped, so the table keeps the first block and the second is released if nobody else holds it. `num_computed_tokens` is clipped to 16. The token-id lists are left for the caller.
5. That block may still match a future prefix. The prefix cache parks it and calls `reclaim` only when it evicts the block.

</details>
