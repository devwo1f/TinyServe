# P1.5 The sampler

## 1. Concept in plain words

The model scores every token in the vocabulary. Sampling turns that row of scores into one id.

Temperature 0 skips the random draw and takes the largest score. That is greedy. A temperature above 0 divides the scores first, which flattens the distribution, then draws.

Top-k keeps only the k largest scores and sets the rest to negative infinity so they cannot be drawn. Top-p (nucleus) sorts what remains, walks down the list adding probabilities, and drops tokens after the running total crosses `top_p`. The token that crosses the line stays. Both filters can be on together: top-k first, then top-p.

A seed has to be per request. If the sampler used the global random generator, one request would change the draws of the next. Each request gets its own `torch.Generator`.

Example: scores `[1, 5, 4]` with top-k 2. Index 0 is removed. The draw is index 1 or 2. Temperature 0 always returns index 1.

## 2. Where it lives in the code

1. `SamplingParams`, `SequenceStatus`, and `Sequence` in `tinyserve/engine/sequence.py`. The field list is spec Section 9. The scheduler later moves a sequence between statuses; this file only holds the data.
2. `sample_token`, `_apply_top_k`, and `_apply_top_p` in `tinyserve/engine/sampler.py`.

## 3. Key tensors and shapes

- logits in: `[batch, vocab]` (a 1-d row is treated as batch 1)
- after top-k / top-p: same shape, rejected positions are `-inf`
- probabilities: `[batch, vocab]`
- chosen ids: `[batch]`, or a scalar when the input was 1-d

## 4. What was measured

No benchmark. CPU unit tests cover greedy, top-k, top-p, a repeated seed, and a batch of two greedy rows.

## 5. Pitfalls hit

`model.generate` on the 1B checkpoint is not this sampler. Its shipped config samples, and `min_new_tokens` bans the end-of-sequence id. Greedy comparisons have to argmax the logits themselves.

Nucleus sampling has to keep the token that crosses `top_p`. Masking before the shift deletes that token and the draw collapses onto the single most likely id.

## 6. Self-check questions

1. What does temperature 0 mean, and why is it not "divide by zero"?
2. Why apply top-k before top-p?
3. Why must the crossing token be kept in nucleus sampling?
4. Why is the seed a `torch.Generator` instead of `torch.manual_seed`?
5. Which `Sequence` field says how much of the cache is already filled?

<details>
<summary>Answers</summary>

1. Temperature 0 is greedy: argmax, no division. Division only happens when the temperature is above 0.
2. Top-k throws away the long tail first, so top-p's cumulative sum is taken over a shorter list. The spec order is temperature, then top-k, then top-p.
3. The threshold is a cumulative probability. The token that pushes the sum over `top_p` is still inside the nucleus. Dropping it leaves less mass than requested.
4. `manual_seed` changes the process-wide generator. Two requests in one batch would then steal random numbers from each other. A generator owned by the request repeats only that request.
5. `num_computed_tokens`.

</details>
