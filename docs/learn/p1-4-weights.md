# P1.4 Loading the checkpoint

## 1. Concept in plain words

A safetensors file is a bag of named tensors and nothing else. Llama 3.2 1B Instruct is one file, `model.safetensors`, stored in bf16. The names are the Hugging Face names (`model.layers.0.self_attn.q_proj.weight`, and so on). TinyServe uses those same names, so loading is a copy, not a translation table.

The 1B checkpoint does not store `lm_head.weight`. The config says the embeddings are tied, which means the LM head is the token embedding. The loader allows that one name to be missing and points the head at `embed_tokens.weight`. Any other missing or extra name is a real mismatch and raises.

The copy goes straight onto the GPU. An 8 GB card cannot hold a CPU staging copy and the live model at the same time, and it also cannot hold Hugging Face's 1B and TinyServe's 1B together. The parity test runs one, saves the tokens, frees the card, then runs the other.

## 2. Where it lives in the code

1. `weight_files` in `tinyserve/model/weights.py` (index, else `model.safetensors`, else every shard).
2. `load_hf_weights` (move the module, `load_file` onto the device, cast dtype, `load_state_dict`, re-tie the head).
3. `tests/gpu/test_llama_parity.py` (greedy loop on both models).
4. `Attention.forward` in `tinyserve/model/llama.py`, the `q_len == 1` branch. The parity test is what showed that branch was wrong.

## 3. Key tensors and shapes

Llama 3.2 1B: vocab 128256, hidden 2048, 16 layers, 32 query heads, 8 KV heads, head dim 64.

- `embed_tokens.weight`: `[128256, 2048]`, also used as `lm_head.weight`
- one attention projection, for example `q_proj.weight`: `[2048, 2048]` (query) or `[512, 2048]` (one KV projection: 8 heads times 64)
- cache written during the parity run: `[16, 1, prompt_len + 64, 8, 64]`

## 4. What was measured

`docs/results/phase1/2026-10-01_p1-4-greedy-parity.json`

bf16, 10 fixed prompts, 64 new tokens, greedy argmax on both sides. The first 32 tokens matched on 10 of 10 prompts. Max absolute difference of the last-prompt-position logits was 0.0. The file was written by the test. The run was dirty: it measured commit `9d09770` plus the uncommitted loader and decode fix.

## 5. Pitfalls hit

The first GPU run matched 8 of 10 prompts. The last-prompt logits already matched (difference 0), so the weights and the prefill were right. Two prompts diverged around token 25 and token 28, on decisions that were nearly tied.

Decode was passing an all-true boolean mask into scaled-dot-product attention. That mask is logically "see every key", but it forces the math kernel. Hugging Face's decode of one new token passes no mask, so the flash kernel runs. Dropping the mask when the query length is 1 brought the match to 10 of 10.

`generate(min_new_tokens=64)` is not plain greedy either: it bans the end-of-sequence id until 64 tokens exist. The test argmaxes each model's own forward pass instead.

## 6. Self-check questions

1. Why is there no `lm_head.weight` in the 1B safetensors file?
2. Why does the loader refuse to stage the tensors on CPU first?
3. How can an all-true attention mask change the chosen token?
4. What does the Section 12 GPU parity test actually require?
5. Why doesn't the reference use `model.generate`?

<details>
<summary>Answers</summary>

1. Llama 3.2 1B ties the embeddings. The LM head matrix is `embed_tokens.weight`. The loader rebinds that pointer after the load.
2. The GPU has 8 GB. A second full copy of the bf16 weights does not fit next to the model.
3. The boolean mask selects a different attention kernel than `is_causal` or an unmasked call. In bf16 those kernels are not bit-identical, so a close argmax can flip after a few dozen tokens.
4. bf16, 64 new tokens, 10 fixed prompts. The first 32 tokens must match on at least 9 prompts, and the max logit difference is reported. This run matched 10 of 10, with max logit difference 0.0. See the result file.
5. `min_new_tokens` suppresses the end-of-sequence id, and the shipped generation config samples. The comparison is argmax of the logits, on both sides.

</details>
