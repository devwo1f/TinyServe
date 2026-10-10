# P4.3 Engine step

## 1. Concept in plain words

One step is one model forward. The scheduler picks who runs. The engine runs those tokens, samples, and lets finished requests go.

Example, two requests, plenty of blocks. The first step prefills both prompts and saves the logit at the last prompt token of each. Nobody has an output token yet. The second step samples both next tokens, appends them, and runs those two new tokens in one forward. When a request hits its token limit, its blocks are freed. The other request keeps decoding.

A decode does not have a token id until it is sampled. The sample uses the logit saved on the previous step. The forward then writes KV for that new token and saves the logit for the step after.

If the pool cannot hold the next token, the scheduler preempts the newest other running request. Its saved logit is dropped, because that KV is gone. It prefills the prompt and the tokens it already generated, then decodes again.

## 2. Where it lives in the code

1. `Engine.generate_tokens` sizes one pool for every prompt in the call and enqueues them.
2. `Engine.step` is the iteration.
3. `_restore_prefixes` shares a cached prompt again after preemption cleared the block table.
4. `Scheduler.schedule` reserves blocks and returns the batch.
5. `_accept_new_token` samples a decode and appends the id. A stop id is not appended.
6. `_forward` calls `prepare_model_input` and `run_paged`, then advances `num_computed_tokens`.
7. `_note_slots` records KV waste while the block tables still exist.
8. `_retire` caches full prompt blocks, frees KV, and calls `Scheduler.finish`.

## 3. Key tensors and shapes

`run_paged` returns logits `[num_seqs, vocab]`, one row per sequence in the forward. The row kept for the next sample is `[vocab]`. The ids, positions, and slots that enter the forward are built in `prepare_model_input`: `input_ids` and `slot_mapping` are `[num_tokens]`.

## 4. What was measured

No result file. The check is token identity on the tiny model: a batched call matches one-at-a-time and Hugging Face greedy ids, a token budget of 4 matches a full prefill, and a 2-block pool matches a roomy pool.

## 5. Pitfalls hit

The model runner refuses a token that is not already on the sequence. The decode path samples and appends, then builds the batch. Doing it the other way around looks at an id that does not exist yet.

A stop token still has a slot, because `schedule` reserved it before the id was known. `truncate` gives that slot back. The token is not written and not emitted.

Preemption deletes the saved logit. Sampling it would score a cache that was just freed.

Two prompts in one call do not see each other's prefix cache. Blocks are cached when a request finishes. A later call is what hits.

## 6. Self-check questions

1. What does one `step` do, in order?
2. Why does a decode sample before the forward?
3. When is a prefill chunk's logit kept?
4. A stop token was scheduled. What happens to its block?
5. Why is the saved logit cleared when a request is preempted?

<details>
<summary>Answers</summary>

1. Restore any cached prefix, schedule, sample decodes, forward the runnable tokens, record slots, then cache and free finished sequences.
2. The scheduler reserved a slot for a token the previous logit already chose. The forward needs that id on the sequence.
3. Only when the chunk lands on the last context token. A mid-prompt logit repeats a token the prompt already contains.
4. `truncate` drops the unused slot. The id is not appended, and the sequence finishes.
5. That logit came from KV that `free` just dropped. The next prefill writes the tokens again and produces a new logit.

</details>
