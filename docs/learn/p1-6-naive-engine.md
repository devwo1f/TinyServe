# P1.6 Generating one request at a time

## 1. Concept in plain words

The model scores the next token. The sampler picks one. That id is appended, and the model runs again with only the new id, reading the keys and values it already stored. That second run is a decode step. The first run, over the whole prompt, is the prefill.

This engine finishes one prompt before it starts the next. The cache is a single dense buffer sized for that prompt plus the tokens it is allowed to write. A second prompt allocates a new buffer. Sharing one buffer would overwrite the first prompt's keys.

Generation stops for two reasons: `max_tokens` is used up, or the sampler draws an id in `stop_token_ids`. The stop id is not part of the output. Temperature 0 is greedy, so the same prompt yields the same ids.

Example: prompt tokens `[10, 20, 30]`, `max_tokens=2`. Prefill writes keys for positions 0, 1, and 2. The first new id is written at position 3, the second at position 4. `num_computed_tokens` is 5.

## 2. Where it lives in the code

1. `Engine.generate` in `tinyserve/engine/engine.py` encodes strings and calls `generate_tokens`.
2. `Engine._run_one` builds a `Sequence`, allocates a `ContiguousKVCache`, prefills, then samples.
3. `sample_token` in `tinyserve/engine/sampler.py`.
4. `Tokenizer.decode` turns the output ids into `GenerationResult.text`.

## 3. Key tensors and shapes

For one request, batch is 1.

- prompt ids: `[1, S]`
- prefill logits: `[1, S, vocab]`, and the next decision uses the last row `[vocab]`
- each decode step: ids `[1, 1]`, logits `[vocab]`
- cache `k` and `v`: `[num_layers, 1, S + max_tokens, num_kv_heads, head_dim]`

`block_table` stays empty. Paging, which fills that list, is Phase 3.

## 4. What was measured

CPU, float32, tiny random model: greedy output ids match a Hugging Face argmax loop exactly, including a second prompt in the same call (its cache is new).

GPU, bf16, Llama 3.2 1B, one prompt, 16 new tokens: `docs/results/phase1/2026-10-02_p1-6-naive-engine.json`. The file says the tokens matched. The recorded completion of "The capital of France is" is " Paris. The capital of Germany is Berlin. The capital of Italy is Rome." The run was dirty: it measured commit `6af5ab9` plus this diff.

## 5. Pitfalls hit

`generate(min_new_tokens=...)` on Hugging Face bans the end-of-sequence id, so it is not the greedy rule this engine uses. The comparison argmaxes each model's forward pass.

`Tokenizer.encode` does not insert a beginning-of-sequence token. The chat template does. Passing a raw chat string through `generate` would shift every position.

## 6. Self-check questions

1. Why does decode feed only the new token, not the whole prompt again?
2. Why is the stop token left out of `output_token_ids`?
3. Why does each prompt get its own cache?
4. What does `num_computed_tokens` count after a request finishes?
5. Why is temperature 0 not a division?

<details>
<summary>Answers</summary>

1. The previous keys and values are already in the cache. Recomputing them would be the same attention with extra matmuls.
2. It is a signal to stop, not part of the answer. Including it would show an end-of-turn marker in the text.
3. The contiguous cache is one dense buffer. The next prompt would overwrite those keys. A scheduler and paged blocks are what make sharing possible later.
4. How many tokens have keys in the cache: the prompt plus every emitted output token. The stop token is not written.
5. Temperature 0 takes `argmax` and returns. Division only happens when the temperature is above 0.

</details>
