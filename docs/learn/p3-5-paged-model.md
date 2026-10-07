# P3.5 Paged model forward

## 1. Concept in plain words

The model no longer needs one dense rectangle per sequence. A step is a flat list of the tokens that are new this step, plus a note of which sequence each token belongs to.

Each sequence already has a block table. The new tokens get slots from that table, the forward writes K and V there, and attention reads the whole context back in logical order. Only the last token of each sequence goes through the LM head, because that is the token the sampler reads.

Example, block size 4. Sequence A prefills five tokens on blocks `[3, 6]`. Sequence B already has three tokens in block `[1]` and decodes one more. The flat batch is six tokens. A's positions are 0, 1, 2, 3, 4. B's position is 3. The block-table tensor is `[[3, 6], [1, -1]]`. The `-1` is padding. Block 0 is a real block, so the pad cannot be 0.

The engine still finishes one request before starting the next. Its pool is only big enough for that request. Sharing one pool across requests is the scheduler's job.

## 2. Where it lives in the code

1. `prepare_model_input` in `tinyserve/engine/model_runner.py` builds the flat tensors. It does not change `num_computed_tokens`.
2. `run_paged` calls `LlamaForCausalLM.forward_paged`.
3. `Attention.forward_paged` in `tinyserve/model/llama.py` projects, applies RoPE, writes the layer with `write_layer_kv`, then calls `paged_attention_flat`.
4. `Engine._run_one` allocates blocks, prefills, then decodes one token at a time, and frees the blocks at the end.
5. `forward` with `ContiguousKVCache` is still there. The dense cache is the reference the paged path is checked against.

## 3. Key tensors and shapes

`input_ids`, `positions`, and `slot_mapping` are `[num_tokens]`.

`query_start_loc` is `[num_seqs + 1]`, a prefix sum of the query lengths. `seq_lens` is `[num_seqs]`, the context length after this step. `block_tables` is `[num_seqs, max_blocks]` int32, trailing `-1`. `logits_indices` is `[num_seqs]`.

Hidden states inside the stack are `[num_tokens, hidden]`. The LM head returns `[num_seqs, vocab]`.

## 4. What was measured

No new result file. A 7-token paged prefill matches the contiguous forward exactly in float32. A chunk over a prefix, and a batch that prefills one sequence while decoding another, match within 1e-5, the same tolerance the contiguous cache already uses against a full forward. The existing 1B engine parity test still passes. Its result file was left as it was.

## 5. Pitfalls hit

The LM head only sees the last token, so a bad mask on an earlier token in the same chunk does not show up in that logit directly. It shows up in the next layer, because that earlier token's hidden state becomes a key. The test compares every logit in the chunk, not only the last one.

`num_computed_tokens` has to move after the forward. If it moved before, the next step would skip the token whose KV was just written, or the block manager would size the table for tokens that are not in the cache yet.

Padding with 0 would make a short sequence read block 0. The pad is `-1`, and it must be trailing.

## 6. Self-check questions

1. Why does a prefill of five tokens and a decode of one token become one `input_ids` tensor?
2. What is `query_start_loc` for query lengths 5 and 1?
3. Why is the block table padded with -1?
4. Why does `prepare_model_input` leave `num_computed_tokens` alone?
5. Why can the engine still match a full forward if it only returns the last logit of the prefill?

<details>
<summary>Answers</summary>

1. The batch is concatenated. `query_start_loc` says where each sequence starts in that list.
2. `[0, 5, 6]`. The first sequence occupies tokens 0 through 4. The second occupies token 5.
3. Block 0 is a real page. A pad of 0 would be gathered as data. `-1` is not a block id, and only trailing pads are allowed.
4. The slots are reserved and then the forward fills them. The count means "KV is already written." The caller sets it after the forward returns.
5. Sampling only needs that logit. The other prefill tokens still have their K and V written, which is what the next decode step reads. A test can put every position in `logits_indices` when it wants the whole row.

</details>
