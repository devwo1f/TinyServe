# P2.1 Workloads

## 1. Concept in plain words

A benchmark is a list of requests that every engine sees. Each request is a prompt plus how many tokens to generate. If one engine stops early at an end-of-sequence token and the other does not, the timing comparison is meaningless, so the output length is fixed and end-of-sequence is ignored later.

Four lists exist. ShareGPT is real chats, cut down to the first human message and the assistant reply after it, and dropped when either side is longer than 1024 tokens. Code is the same idea with a programming prompt and its solution. Shared-prefix is many requests that start with the same tokens, which is what prefix caching is for. Synthetic is random token ids of a length you choose, so a sweep can hold the shape still.

The seed does not shuffle with Python's random module. It hashes each sample and sorts. The same file and the same seed produce the same subset.

Example: prefix tokens `[1, 2, 3]` and suffixes `[4]` and `[5, 6]`. The two prompts are `[1, 2, 3, 4]` and `[1, 2, 3, 5, 6]`. The first three ids match, so a prefix cache can reuse them.

## 2. Where it lives in the code

1. `Sample` in `bench/datasets.py`.
2. `load_sharegpt` and `_sharegpt_pairs`.
3. `load_code` and `_read_records` (JSON list or JSONL).
4. `shared_prefix_workload`.
5. `synthetic_workload`.
6. `_filter_lengths`, then `take_subset`.

## 3. Key tensors and shapes

There are no model tensors here. Each `Sample` carries `prompt_token_ids`, a 1-D list whose length is the prompt length the filter checked, and `output_len`, an int.

## 4. What was measured

No benchmark. Unit tests on small fixtures check the filter, the subset, and the four workload shapes. The real ShareGPT and HumanEval files are downloaded and gitignored; CI does not need them.

## 5. Pitfalls hit

Encoding `prefix + suffix` as one string can merge the boundary into a different token. The shared-prefix workload concatenates `encode(prefix)` and `encode(suffix)` so the shared span is actually shared.

The Hugging Face HumanEval repo is parquet. Reading it would add a package. The download script fetches the official JSONL instead (D-009).

## 6. Self-check questions

1. Why is the output length stored on the sample instead of letting the model stop?
2. Why does ShareGPT keep only the first human/assistant pair?
3. Why hash-sort the subset instead of `random.shuffle`?
4. Why can a string prefix fail to be a token prefix?
5. What does a synthetic sample use for its prompt string?

<details>
<summary>Answers</summary>

1. Two engines must do the same amount of work. End-of-sequence would stop one of them early. The benchmark sets `max_tokens` to `output_len` and ignores that token.
2. One request is one prompt and one target length. Later turns would be a second request, or a longer prompt that the filter is no longer describing.
3. `shuffle` depends on the Python version's algorithm. The hash of the sample contents does not, so a seed means the same rows.
4. Byte-level BPE can join the last characters of the prefix with the first characters of the suffix into one token. Concatenating the two id lists avoids that join.
5. It is empty. The ids are random integers, not text. The benchmark sends `prompt_token_ids`.

</details>
