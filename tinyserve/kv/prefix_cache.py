"""Hash-chain prefix cache for full KV blocks.

A later request with the same prompt prefix can reuse the KV that an earlier
request already computed. Only full blocks are stored. The hash of a block
includes the parent block's hash, so a change early in the prompt cannot hit
a later block that happens to hold the same tokens.

The last prompt token is always recomputed. Sampling needs a logit, and that
token's block is not reused when the token sits inside it.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict


def chain_hash(parent_hash: int, token_ids: tuple[int, ...]) -> int:
    """64-bit hash of one full block and the prefix that came before it.

    ``parent_hash`` is 0 for the first block. Python's built-in hash is salted
    per process, so two runs would not agree on a value a test can write down.
    This digest stays inside the process; it is not a security hash.
    """
    if parent_hash < 0:
        raise ValueError("parent_hash must be >= 0")
    digest = hashlib.sha256()
    digest.update(parent_hash.to_bytes(8, "little"))
    digest.update(len(token_ids).to_bytes(4, "little"))
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return int.from_bytes(digest.digest()[:8], "little")


class PrefixCache:
    """Map a prompt's full blocks onto physical block ids, and evict the cold ones.

    A cached block with ref count 0 sits on an LRU list. ``park`` adds it.
    ``touch`` takes it off when a sequence starts using it. ``evict_lru``
    forgets the oldest id; the block manager then returns that id to the
    free list.
    """

    def __init__(self, block_size: int):
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self._by_hash: dict[int, int] = {}
        self._by_block: dict[int, int] = {}
        # Oldest at the front. Only ref-count-0 blocks are here.
        self._lru: OrderedDict[int, None] = OrderedDict()

    def is_cached(self, block_id: int) -> bool:
        """True when this physical block still holds a prefix the cache may reuse."""
        return block_id in self._by_block

    def cache_prompt(self, block_table: list[int], prompt_ids: list[int]) -> None:
        """Remember every full block of ``prompt_ids``.

        The block table is in logical order, so index ``i`` holds tokens
        ``[i * block_size, (i + 1) * block_size)``. A partial last block is
        skipped. If the hash is already stored under another physical block,
        that older block stays; this one is left uncached so it can be freed.
        """
        full_blocks = len(prompt_ids) // self.block_size
        if full_blocks > len(block_table):
            raise ValueError("block table is shorter than the prompt")
        parent = 0
        for index in range(full_blocks):
            start = index * self.block_size
            tokens = tuple(prompt_ids[start : start + self.block_size])
            digest = chain_hash(parent, tokens)
            block_id = block_table[index]
            current = self._by_hash.get(digest)
            if current is None:
                self._by_hash[digest] = block_id
                self._by_block[block_id] = digest
            parent = digest

    def match(self, prompt_ids: list[int]) -> list[int]:
        """Physical block ids for the cached prefix, in logical order.

        The walk stops at the first miss. It also stops before the last
        prompt token, so the caller still has a token to run for logits.
        A block that contains that last token is not returned, even when
        the block itself is cached.
        """
        if not prompt_ids:
            return []
        limit = (len(prompt_ids) - 1) // self.block_size
        parent = 0
        found: list[int] = []
        for index in range(limit):
            start = index * self.block_size
            tokens = tuple(prompt_ids[start : start + self.block_size])
            digest = chain_hash(parent, tokens)
            block_id = self._by_hash.get(digest)
            if block_id is None:
                break
            found.append(block_id)
            parent = digest
        return found

    def park(self, block_id: int) -> None:
        """The last sequence let go of a cached block. It can be evicted."""
        if block_id not in self._by_block:
            raise RuntimeError("parked a block the prefix cache does not own")
        self._lru[block_id] = None
        self._lru.move_to_end(block_id)

    def touch(self, block_id: int) -> None:
        """A sequence is using this block, so it leaves the evictable list."""
        self._lru.pop(block_id, None)

    def evict_lru(self) -> int | None:
        """Drop the oldest unused cached block and return its id.

        The caller reclaims that id onto the free list. ``None`` means every
        cached block is still in use.
        """
        if not self._lru:
            return None
        block_id, _ = self._lru.popitem(last=False)
        digest = self._by_block.pop(block_id)
        if self._by_hash.get(digest) == block_id:
            del self._by_hash[digest]
        return block_id
