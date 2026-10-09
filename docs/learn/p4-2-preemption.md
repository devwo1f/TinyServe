# P4.2 Preemption

## 1. Concept in plain words

The block pool is finite. A request that needs one more block cannot take it from nowhere. The scheduler drops the KV of the newest other request that is already running, puts those blocks back on the free list, and runs the request that was blocked.

The dropped request is not cancelled. Its generated tokens stay. `num_computed_tokens` goes back to 0, so the next time it runs, the prefill writes the prompt and those tokens again. Then it continues decoding.

Example, two blocks of 4. Request A arrived first and holds one full block. Request B arrived later and holds the other. A needs a fifth token, which opens a second block. The free list is empty, so B is preempted. A takes B's block. B waits, then prefills again after A finishes and releases its blocks.

A request that is alone is not preempted. Restarting it would free the block and then need that same block back. Nothing else is waiting to use it.

## 2. Where it lives in the code

1. `Scheduler._make_room` in `tinyserve/engine/scheduler.py` loops while `can_allocate` is false.
2. `_newest` picks the latest `arrival_time` among running requests this step has not already reserved.
3. `_preempt` calls `BlockManager.free`, zeroes `num_computed_tokens`, sets status `PREEMPTED`, and puts the request at the front of the waiting queue.
4. `_context_len` is the prompt plus tokens already generated. A preempted request prefills that whole span before the next new token.
5. `ScheduledBatch.preempted` is who lost KV on this call.

The engine does not call this yet. Finished requests are still freed by the caller. That loop is P4.3.

## 3. Key tensors and shapes

No tensors. Preemption clears `block_table` and leaves `output_token_ids` as they were.

## 4. What was measured

No result file. Two requests with prompt 4 and 4 new tokens each finish on a pool of 2 blocks of size 4, and their token ids match a pool of 8 blocks.

## 5. Pitfalls hit

Preempting the only running request livelocks. The next step starts it again, it fills the pool, and it is preempted again. The scheduler refuses that victim.

The token budget is not a reason to preempt. Freeing KV does not make the budget larger. A prompt that does not fit in the budget stays at the head of the queue.

Generated tokens have to be prefilled again. Resetting `num_computed_tokens` and leaving them off the context would make the next sample ignore them.

## 6. Self-check questions

1. Who is preempted when two requests are running and the pool is full?
2. What happens to tokens that request had already generated?
3. Why is a lone request left running when it needs one more block?
4. Why does the next prefill include the output tokens?
5. What does the caller still have to do after a request hits `max_tokens`?

<details>
<summary>Answers</summary>

1. The one with the later `arrival_time`, unless this step already reserved blocks for it. The older request keeps its KV and takes the freed block.
2. They stay on `output_token_ids`. The block table is cleared and `num_computed_tokens` is 0. Status is `PREEMPTED`, and the request sits at the front of the waiting queue.
3. It is the only source of blocks. Freeing it and recomputing it asks for those blocks again. No other request makes progress.
4. Their KV was discarded. The next sample has to see them, so the prefill runs the prompt and the saved output before it draws a new token.
5. Free its blocks and take it off the running queue. `schedule` does not do that. P4.3's step loop will.

</details>
