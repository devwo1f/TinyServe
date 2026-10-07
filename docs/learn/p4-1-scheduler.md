# P4.1 Scheduler

## 1. Concept in plain words

A step has a token budget and a cap on how many requests are in flight. Decode tokens are one each, and they are chosen before any prompt. A prompt then fills whatever budget is left. If the prompt is longer than that leftover, it runs as a chunk and continues on a later step.

Example, budget 4, block size 16. Request A is already generating. Request B has a 10-token prompt and is waiting. The step runs A's one decode token, then the first 2 tokens of B. B's other 8 tokens wait. Next step, if A is still generating, A goes again, and B continues before any newer request.

The scheduler reserves the KV blocks for those tokens. It does not mark them computed. The forward does that, and the caller adds the chunk length to `num_computed_tokens` afterwards.

## 2. Where it lives in the code

1. `ScheduledSeq` and `ScheduledBatch` in `tinyserve/engine/scheduler.py`.
2. `Scheduler.add` puts a request on the waiting deque.
3. `Scheduler.schedule` walks running decodes, then running prefills, then the head of the waiting queue.
4. `BlockManager.allocate` reserves the blocks inside that call.
5. `_rotate_decodes` moves decodes that ran to the back of the running queue.

The engine does not call this yet. That loop is P4.3. Preemption is P4.2, so `preempted` is always empty here.

## 3. Key tensors and shapes

No tensors. The batch is a list of sequences plus an integer token count. A decode entry is `num_new_tokens == 1` and `is_prefill` false. A prefill entry's `num_new_tokens` is the chunk length.

## 4. What was measured

No result file. The tests are the check: budget sums, chunk boundaries 4/4/2, the sequence cap, and a one-block pool that stops at 4 tokens.

## 5. Pitfalls hit

`allocate` sizes the block table from `num_computed_tokens + num_new_tokens`. If `num_computed_tokens` is still the old value, a second `schedule` before the forward reserves those tokens again. The caller has to add the chunk length after the forward.

A waiting request that does not fit stops the queue. The shorter request behind it stays waiting too. That is deliberate: skipping it would admit work the head of the queue cannot start. When the pool itself is empty, the step is empty and nobody is preempted yet.

## 6. Self-check questions

1. Why do decodes run before prefills?
2. Budget 4, prompt 10, chunking on. What are the prefill chunk sizes?
3. What does `max_num_seqs` limit?
4. Why is `num_computed_tokens` unchanged when `schedule` returns?
5. A decode needs a new block and the free list is empty. What does this scheduler do?

<details>
<summary>Answers</summary>

1. A long prompt would otherwise fill the whole step, and requests that are already generating would stall. Decode tokens are one each, so they are cheap to place first.
2. 4, then 4, then 2. The next step after the prompt is a single decode token.
3. How many sequences may be in the running queue. A step does not admit a new request once that many are already in flight. A request that is already running can still take a chunk.
4. That count means KV has been written. `schedule` only reserves slots. The forward writes them, and the caller adds `num_new_tokens`.
5. It leaves the sequence running and schedules nothing for it. `preempted` is empty. Freeing someone else's blocks is P4.2.

</details>
