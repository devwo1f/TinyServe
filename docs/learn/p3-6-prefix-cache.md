# P3.6 Prefix cache

## 1. Concept in plain words

Two prompts that start with the same tokens already have the same K and V for that prefix. The engine can keep those pages and skip recomputing them.

Only a full block is reusable. The hash of a block includes the hash of the block before it, so a later block cannot hit just because its own tokens match. The last prompt token is always run again. Sampling needs a logit, and a block that still contains that token is not handed out.

Example, block size 4. Prompt A is eight tokens and fills blocks 0 and 1. Prompt B is those eight tokens plus one more. B may reuse both blocks (eight tokens) and prefill only the ninth. A second copy of A may reuse only block 0. The last token of A sits inside block 1, so that block stays out of the match even though it is cached. If the free list is empty, the oldest unused cached block is forgotten and becomes free again.

## 2. Where it lives in the code

1. `chain_hash` and `PrefixCache` in `tinyserve/kv/prefix_cache.py`.
2. `BlockManager.share` in `tinyserve/kv/block_manager.py` can attach a parked block. `free` still parks a cached block instead of returning it.
3. `Engine._ensure_pool` in `tinyserve/engine/engine.py` builds one pool for the prompts in the call.
4. `_take_cached_prefix` matches, shares, and sets `num_computed_tokens` to the reused length.
5. `_reserve` evicts the LRU cached block when `can_allocate` is false, then allocates the tail.
6. After the request, `cache_prompt` records the full blocks and `free` parks them.
7. `bench/prefix_cache.py` runs the shared-prefix prompts twice and writes the result file.

## 3. Key tensors and shapes

The cache stores no extra K/V tensor. It stores three maps: hash to block id, block id to hash, and an LRU of block ids whose ref count is 0.

A reused prefix is still the existing paged K/V. The prefill `input_ids` for the tail is `[remaining]`, with positions starting at the first uncached token. `num_cached_prompt_tokens` on the result is an int, not a tensor.

## 4. What was measured

`docs/results/phase3/2026-10-07_p3-6-prefix-cache.jsonl`. Four Llama-3.2-1B prompts, 100 tokens each, output length 8, block size 16. Computed prompt tokens fell from 400 to 112. The cached count is 288. Greedy token ids matched the uncached run. TTFT percentiles are on the summary line. The median barely moved. The off run is first, so its first request pays the cold start, and that is what pulls p90 and p99 up.

## 5. Pitfalls hit

Python's built-in `hash` is salted per process. The chain uses SHA-256 so a test can depend on the parent link, not on a value that changes every launch.

A prompt whose length is an exact multiple of the block size does not reuse its last block. The last token lives in that block. A longer prompt that continues past it can reuse the block. On this measurement the shared text is 97 tokens and each full prompt is 100, so six blocks (96 tokens) are reusable. The 97th prefix token sits in the tail with the suffix.

The first request in a call is always a miss. It fills the cache. Later requests in the same call see it. A later call that needs a bigger pool throws the pages away and starts empty.

## 6. Self-check questions

1. Why does the hash include the parent block?
2. Why is the last prompt token never served from the cache?
3. Prompt length 8, block size 4. How many blocks can a second copy of that prompt reuse?
4. Where does a cached block go when its ref count hits 0?
5. Why can two requests with the same tokens still disagree if one of them changed an early token?

<details>
<summary>Answers</summary>

1. Otherwise a later block with the same tokens would hit even when the prefix before it was different. The parent hash makes the id depend on the whole prefix.
2. The sampler needs a logit. That logit is the forward of the last prompt token. A block that still contains it is left for the prefill.
3. One. The second block holds the last token, so the match stops after the first block. A 9-token prompt can reuse both.
4. Onto the LRU list, not the free list. The pages stay valid until `evict_lru` forgets the hash and the block manager reclaims the id.
5. The later block's hash was computed with a different parent. The walk stops at the first miss, so nothing after the change is reused.

</details>
