# P1.3 The Llama forward pass

## 1. Concept in plain words

A Llama layer is a residual block done twice. First, normalize the hidden state, run attention, and add the result back onto the original state. Then normalize again, run the MLP, and add that back too. The add-back is why a deep stack does not wash the token away.

Attention here is grouped-query attention. The tiny test model has 4 query heads and 2 key/value heads, so each KV head is read by 2 query heads. The cache stores only the KV heads. That is the whole memory win: the cache does not grow with the query-head count.

The MLP is SwiGLU. One projection is passed through SiLU and multiplied by a second projection, then a third projection brings it back to the hidden size. Two "up" matrices, gated against each other, instead of one.

The KV cache in this phase is a dense block, one slot per position up to a fixed maximum. A 6-token sequence still reserves 6 slots even after 4 tokens. Paging (Phase 3) exists to stop paying for the empty slots. The dense cache is here so the math can be checked first.

Example: prefill tokens 0,1,2,3, then feed token 4 alone. The new query must see keys 0 through 4, not only key 0. On this PyTorch build, `is_causal=True` is wrong when the key list is longer than the query and the query is more than one token, so that case builds an explicit mask. A single new token can see every key, so it passes no mask.

## 2. Where it lives in the code

1. `LlamaModelConfig.from_json` in `tinyserve/model/llama.py`.
2. `RMSNorm.forward`, then `Attention.forward` (`q_proj`, `k_proj`, `v_proj`, `apply_rotary`, `repeat_kv`, SDPA, `o_proj`).
3. `MLP.forward`.
4. `DecoderLayer.forward` (the two residuals).
5. `LlamaModel.forward` (embed, RoPE cos/sin, layers, final norm, cache length).
6. `LlamaForCausalLM.forward` (LM head; tied to the embedding when the config says so).
7. `ContiguousKVCache.write`.

## 3. Key tensors and shapes

For the tiny model, B=1, H=64, Nq=4, Nk=2, D=16, I=128.

- `input_ids`: `[B, S]`
- embedding: `[B, S, H]`
- query after projection: `[B, Nq, S, D]`
- key and value after projection: `[B, Nk, S, D]`
- after `repeat_kv`: key and value are `[B, Nq, Sk, D]`
- cache `k` and `v`: `[num_layers, B, max_len, Nk, D]`
- logits: `[B, S, vocab]`

## 4. What was measured

No benchmark. CPU float32 logits match Hugging Face within 1e-4 absolute on the tiny random model, including tied embeddings. A prefill-plus-decode run matches one forward of the whole sequence.

## 5. Pitfalls hit

`scaled_dot_product_attention(..., is_causal=True)` on this PyTorch, when the key sequence is longer than the query, lets query position 0 see only key position 0. A query of several tokens uses an explicit boolean mask so those queries sit at the end of the cache. A single new token can see every key, so it passes no mask: an all-true mask forces the math kernel, and bf16 decode then drifts off the flash path Hugging Face uses. Prefill, where the lengths match, still uses `is_causal=True`.

## 6. Self-check questions

1. Why does grouped-query attention shrink the KV cache?
2. What does the SiLU gate multiply, and why are there three linear layers in the MLP?
3. Why is the residual added after attention and again after the MLP?
4. Why can't decode use `is_causal=True` on this machine when the cache is longer than the new query?
5. What does tying the embeddings mean for the LM head weight?

<details>
<summary>Answers</summary>

1. The cache stores KV heads, not query heads. With 4 query heads and 2 KV heads, the cache is half as big as full multi-head attention. Llama 3.1 8B is 32 query heads and 8 KV heads, so the cache is a quarter of full attention.
2. `silu(gate(x)) * up(x)`, then `down`. The gate and the up projection are the two paths that get multiplied; down maps back to the hidden size.
3. Each sublayer computes a change and adds it to its input. The original hidden state is still there for the next layer to read.
4. That flag builds a mask where query 0 may only see key 0, even if five keys are present. A single new query must see the whole cache, which is what an unmasked attention does. A longer query that is still shorter than the cache needs an explicit mask that puts the queries at the end of the keys.
5. `lm_head.weight` is the same matrix as `embed_tokens.weight`. The model does not learn a second copy. Llama 3.2 1B is tied; the tiny fixture is not.

</details>
