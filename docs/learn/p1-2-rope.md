# P1.2 Rotary embeddings and Llama 3 frequency scaling

## 1. Concept in plain words

Attention by itself does not know which token came first. RoPE puts that information into the queries and keys: each pair of channels is rotated by an angle that grows with the position. A token at position 5 is rotated five times as far as a token at position 1, for a given frequency.

Llama 3 was trained with a context of 8192 tokens but is asked to run much further (the 1B model config says 131072). The trick is to change only the slow rotations. A wavelength is how many positions it takes for that frequency to complete a full turn. Short wavelengths (high frequency) stay as trained. Long wavelengths are divided by a factor (8 for the tiny fixture, 32 for Llama 3.2 1B), which stretches them. Wavelengths in between are a blend of the two, so there is no sudden jump.

Example: with the tiny fixture, `original_max_position_embeddings` is 8192 and `factor` is 8. Position 8193 is one past the original context. If the long waves were not stretched, the model would be using angles it never saw in training.

## 2. Where it lives in the code

1. `tinyserve/model/rope.py`: `RopeScaling.from_config` reads the checkpoint's `rope_scaling` dict.
2. `compute_inv_freq` builds the frequencies, shape `[head_dim / 2]`.
3. `rotary_cos_sin` turns positions into cos and sin, shape `[batch, seq, head_dim]`.
4. `apply_rotary` rotates queries and keys.

## 3. Key tensors and shapes

- `inv_freq`: `[head_dim / 2]`. For the tiny model, head_dim is 16, so this has 8 frequencies.
- `position_ids`: `[batch, seq]`.
- `cos`, `sin`: `[batch, seq, head_dim]`. Each angle is repeated on both halves so it matches `rotate_half`.
- After `apply_rotary`, query is still `[batch, num_heads, seq, head_dim]` and key is `[batch, num_kv_heads, seq, head_dim]`.

## 4. What was measured

No benchmark. The test checks numerical agreement with Hugging Face, not speed.

## 5. Pitfalls hit

The medium band has to use the same closed interval as the reference (`wavelength >= high` and `wavelength <= low`). An off-by-one at the boundary would still pass a test that only checks easy positions, which is why the test includes 8191, 8192, and 8193.

## 6. Self-check questions

1. What problem does RoPE solve that a bag of token vectors does not?
2. Why is the same angle written into both halves of the head dimension?
3. In Llama 3 scaling, which frequencies change, and which stay the same?
4. Why is a wrong RoPE implementation dangerous even when short prompts look fine?
5. What is the shape of `cos` for a batch of 2 sequences of length 5 and head_dim 16?

<details>
<summary>Answers</summary>

1. Order. Without a position signal, attention treats a permutation of the same tokens as the same input.
2. `rotate_half` swaps the two halves and negates one of them. The angle has to be present on both halves or the rotation mixes the wrong channels.
3. Long wavelengths (low frequency) are divided by `factor`. Short wavelengths stay put. The band between `high_freq_factor` and `low_freq_factor` is blended.
4. Short prompts never reach the scaled region, so greedy text can look normal while anything past the original context (8192 for Llama 3) is quietly wrong.
5. `[2, 5, 16]`.

</details>
