"""Who owns each KV block.

The cache tensor is one pool. This module hands out block ids, remembers how
many sequences still need each block, and takes a block back when the last
user lets go. Prefix caching (P3.6) will call `share` for a reused full block
and `reclaim` when an unreferenced cached block is evicted. Until then a
block at ref count 0 returns to the free list, unless the caller says it is
cached.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from tinyserve.engine.sequence import Sequence


def blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    """How many blocks cover `num_tokens`. Zero tokens hold no block.

    A length that is an exact multiple of `block_size` fills the last block.
    One token past that multiple needs another block.
    """
    if num_tokens < 0:
        raise ValueError("num_tokens must be >= 0")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_tokens == 0:
        return 0
    return (num_tokens + block_size - 1) // block_size


class BlockManager:
    """Free list and reference counts for a fixed pool of block ids.

    `allocate` only appends ids. It does not mark tokens computed: the
    scheduler writes `num_computed_tokens` after the forward that fills them.
    The count used here is `num_computed_tokens + num_new_tokens`, which is
    how many tokens must have a slot when that forward runs.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        *,
        is_cached: Callable[[int], bool] | None = None,
        park_cached: Callable[[int], None] | None = None,
    ):
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: deque[int] = deque(range(num_blocks))
        self._free_ids = set(self._free)
        self._ref_count = [0] * num_blocks
        self._is_cached = is_cached or (lambda _block_id: False)
        self._park_cached = park_cached

    def num_free_blocks(self) -> int:
        return len(self._free)

    def ref_count_of(self, block_id: int) -> int:
        self._check_id(block_id)
        return self._ref_count[block_id]

    def can_allocate(self, seq: Sequence, num_new_tokens: int) -> bool:
        """True when the free list can cover the new tokens, including a partial last block."""
        return self._extra_blocks(seq, num_new_tokens) <= len(self._free)

    def allocate(self, seq: Sequence, num_new_tokens: int) -> None:
        """Append fresh block ids so `seq` can hold `num_new_tokens` more tokens."""
        extra = self._extra_blocks(seq, num_new_tokens)
        if extra > len(self._free):
            raise RuntimeError(f"need {extra} KV blocks but only {len(self._free)} are free")
        for _ in range(extra):
            seq.block_table.append(self._take_free())

    def share(self, seq: Sequence, block_id: int) -> None:
        """A second sequence reuses `block_id` instead of taking a free one.

        The block stays out of the free list until every sequence that holds
        it has freed or truncated it.
        """
        self._check_id(block_id)
        if self._ref_count[block_id] <= 0:
            raise ValueError("cannot share a free block")
        if block_id in seq.block_table:
            raise ValueError("sequence already holds this block")
        self._ref_count[block_id] += 1
        seq.block_table.append(block_id)

    def free(self, seq: Sequence) -> None:
        """Drop every block the sequence holds. Shared blocks stay allocated."""
        for block_id in seq.block_table:
            self._decref(block_id)
        seq.block_table.clear()

    def truncate(self, seq: Sequence, new_length: int) -> None:
        """Free blocks that sit entirely past `new_length` tokens.

        Speculative rollback uses this. A length that still touches a block
        keeps that block: 16 tokens with block size 16 keep one block, 17
        keep two. `num_computed_tokens` is clipped so it cannot point past
        the tokens that still have slots. Token id lists are the caller's.
        """
        if new_length < 0:
            raise ValueError("new_length must be >= 0")
        keep = blocks_for_tokens(new_length, self.block_size)
        if keep > len(seq.block_table):
            raise ValueError("truncate cannot grow the block table")
        dropped = seq.block_table[keep:]
        del seq.block_table[keep:]
        for block_id in dropped:
            self._decref(block_id)
        if seq.num_computed_tokens > new_length:
            seq.num_computed_tokens = new_length

    def reclaim(self, block_id: int) -> None:
        """Put an unreferenced cached block back on the free list.

        The prefix cache calls this on eviction. A block that still has
        readers, or that is already free, is a bug.
        """
        self._check_id(block_id)
        if self._ref_count[block_id] != 0:
            raise RuntimeError("cannot reclaim a block that is still referenced")
        self._push_free(block_id)

    def _extra_blocks(self, seq: Sequence, num_new_tokens: int) -> int:
        if num_new_tokens < 0:
            raise ValueError("num_new_tokens must be >= 0")
        need = blocks_for_tokens(seq.num_computed_tokens + num_new_tokens, self.block_size)
        return max(0, need - len(seq.block_table))

    def _take_free(self) -> int:
        block_id = self._free.popleft()
        self._free_ids.remove(block_id)
        if self._ref_count[block_id] != 0:
            raise RuntimeError("free list contained a referenced block")
        self._ref_count[block_id] = 1
        return block_id

    def _decref(self, block_id: int) -> None:
        self._ref_count[block_id] -= 1
        if self._ref_count[block_id] < 0:
            raise RuntimeError(f"block {block_id} ref count went negative")
        if self._ref_count[block_id] != 0:
            return
        if self._is_cached(block_id):
            if self._park_cached is None:
                raise RuntimeError("cached block reached ref count 0 with nowhere to park it")
            self._park_cached(block_id)
            return
        self._push_free(block_id)

    def _push_free(self, block_id: int) -> None:
        if block_id in self._free_ids:
            raise RuntimeError(f"block {block_id} is already free")
        self._free.append(block_id)
        self._free_ids.add(block_id)

    def _check_id(self, block_id: int) -> None:
        if not 0 <= block_id < self.num_blocks:
            raise ValueError(f"block id {block_id} is outside 0..{self.num_blocks - 1}")
