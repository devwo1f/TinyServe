# P3.7 KV memory waste

## 1. Concept in plain words

A dense KV cache reserves the maximum length for a request before the first token exists. Those empty slots are waste. Paging reserves a block only when a token needs it. The leftover is the unused tail of the last block, at most `block_size - 1` tokens.

Waste is `(allocated - used) / allocated`, taken after each step. Used slots are tokens whose K and V are written.

Example, block size 16, serving limit 4096. A request of 17 tokens uses 17 slots. Paging holds two blocks, 32 slots, so waste is 15/32. The contiguous reservation is still 4096 slots.

Eight requests held at once, at their full lengths, use 516 slots. Paging holds 576. The contiguous reservation holds 8 × 4096 = 32768.

## 2. Where it lives in the code

1. `kv_memory_waste` and `sample_slots` in `tinyserve/kv/waste.py`.
2. `request_step_samples` allocates with `BlockManager` the way one engine request steps: one prefill, then one token at a time.
3. `concurrent_slot_sample` keeps every length allocated together.
4. `Engine._slot_sample` in `tinyserve/engine/engine.py` records a sample after `num_computed_tokens` moves, and before `free` clears the table. It runs only when `record_kv_waste` is set.
5. `bench/kv_waste.py` runs the eight lengths and writes the result file.

## 3. Key tensors and shapes

This task does not allocate a new tensor. It counts integers. A block table `[0, 1]` with block size 16 is 32 paged slots. `max_len` is one integer per live sequence, not a row in the cache.

## 4. What was measured

`docs/results/phase3/2026-10-07_p3-7-kv-waste.jsonl`. Eighty steps on the tiny model, block size 16, contiguous max length 4096.

One request at a time, paged waste p50 is 0.05610119047619048 and contiguous waste p50 is 0.9754638671875. With all eight requests live at full length: 516 used, 576 paged, 32768 contiguous. Paged waste 0.10416666666666667. Contiguous waste 0.9842529296875.

The same lengths on Llama 3.2 1B would record the same slot counts. The weights do not decide how many blocks a length needs.

## 5. Pitfalls hit

The one-at-a-time p99 paged waste is 0.8881249999999996 because a one-token step still holds a whole block of 16. That tail is real. It is not the serving comparison. The serving comparison is the concurrent snapshot, where the short requests are already at their full length and the dense cache is still charging 4096 each.

Samples have to be taken before `free`. After `free` the block table is empty and the used count is not.

`max_len` is the config's 4096, not the tiny fixture's context of 512. The requests fit in 512. The contiguous side still charges the serving limit.

## 6. Self-check questions

1. What is the waste formula?
2. A sequence has 17 tokens and the block size is 16. How many paged slots does it hold?
3. Why can contiguous waste stay near 1 while paged waste is about 0.1?
4. Why is a one-token request a high paged-waste step?
5. Why does this measurement not load Llama 3.2 1B?

<details>
<summary>Answers</summary>

1. `(allocated slots - used slots) / allocated slots`. Used slots are tokens with KV written. Zero allocated slots is waste 0.
2. 32. Two blocks. Fifteen of those slots are the unused tail.
3. Contiguous allocation charges `max_len` (4096) per live sequence from the first step. Paging charges only the blocks the current tokens need. On the concurrent snapshot that is 32768 versus 576, for 516 tokens actually stored.
4. The block is the unit of allocation. One token occupies one block, so 15 of 16 slots are empty. A longer sequence fills more of its blocks, and the concurrent snapshot is that later state.
5. The slot count is `ceil(length / block_size) * block_size`. The model weights do not change it. The tiny model is what the engine stepped so the rows are real step samples.

</details>
