"""Block-paged KV cache with refcounted copy-on-write prefix sharing.

This mirrors the memory-management mechanics of vLLM's PagedAttention /
Automatic Prefix Caching closely enough to reproduce the *scheduling
consequences* we care about: which requests get cheap prefills because their
prefix is already resident, and which requests get evicted or preempted when
the block pool saturates.

It is deliberately a reference model, not a reimplementation of vLLM. See
README.md ("What is simulated, what is measured") for the exact boundary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


def block_hash(parent_hash: Optional[str], tokens: Sequence[int]) -> str:
    """Chained block hash, as used for prefix-cache keys.

    The hash of a block depends on the hash of every block before it, so a
    match on block k implies a match on blocks 0..k. This chaining is what
    makes prefix reuse a *prefix* property and is also what makes the cache
    observable as a monotone signal (the property the side-channel literature
    exploits; here we care about its scheduling effect).
    """
    h = hashlib.blake2b(digest_size=16)
    if parent_hash is not None:
        h.update(parent_hash.encode())
    h.update(b"|")
    h.update(",".join(str(t) for t in tokens).encode())
    return h.hexdigest()


@dataclass
class Block:
    block_id: int
    hash: Optional[str] = None      # None for non-cacheable (partial/decode) blocks
    ref_count: int = 0
    last_used: float = 0.0


@dataclass
class AllocationResult:
    ok: bool
    cached_blocks: int = 0          # blocks satisfied from the prefix cache
    computed_blocks: int = 0        # blocks that must be prefilled
    blocks_short: int = 0           # how many blocks we could not obtain


class BlockAllocator:
    """Fixed pool of KV blocks with LRU eviction of unreferenced cached blocks.

    Invariants (asserted in tests):
      * sum(ref_count over cached blocks) never counts a block twice per request
      * a block with ref_count > 0 is never evicted
      * free_blocks + allocated_blocks == total_blocks at every quiescent point
    """

    def __init__(self, total_blocks: int, block_size: int = 16):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self._blocks: Dict[int, Block] = {i: Block(i) for i in range(total_blocks)}
        self._free: List[int] = list(range(total_blocks))
        self._by_hash: Dict[str, int] = {}          # hash -> block_id (cached, may be free)
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # -- introspection -----------------------------------------------------
    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def utilisation(self) -> float:
        return 1.0 - self.num_free / self.total_blocks

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    # -- core --------------------------------------------------------------
    def block_hashes(self, token_ids: Sequence[int]) -> List[str]:
        """Chained hashes for every *full* block of a token sequence."""
        hashes: List[str] = []
        parent: Optional[str] = None
        n_full = len(token_ids) // self.block_size
        for i in range(n_full):
            chunk = token_ids[i * self.block_size:(i + 1) * self.block_size]
            parent = block_hash(parent, chunk)
            hashes.append(parent)
        return hashes

    def probe(self, token_ids: Sequence[int]) -> int:
        """Longest cached block prefix, without mutating refcounts.

        Used by latency-predicting policies; does not count as a cache access.
        """
        n = 0
        for h in self.block_hashes(token_ids):
            if h in self._by_hash:
                n += 1
            else:
                break
        return n

    def allocate(
        self, token_ids: Sequence[int], decode_blocks: int, now: float
    ) -> Tuple[AllocationResult, List[int]]:
        """Reserve blocks for a request's prompt + expected decode growth."""
        hashes = self.block_hashes(token_ids)
        held: List[int] = []

        # 1. Reuse the longest resident prefix (copy-on-write: bump refcount).
        cached = 0
        for h in hashes:
            bid = self._by_hash.get(h)
            if bid is None:
                break
            blk = self._blocks[bid]
            if blk.ref_count == 0:
                # Resurrect a cached-but-free block: it leaves the free list.
                self._free.remove(bid)
            blk.ref_count += 1
            blk.last_used = now
            held.append(bid)
            cached += 1

        self.hits += cached
        self.misses += len(hashes) - cached

        # 2. Blocks that must actually be computed, plus room to decode into.
        total_blocks_needed = (len(token_ids) + self.block_size - 1) // self.block_size
        to_compute = total_blocks_needed - cached
        need = to_compute + decode_blocks

        if need > self.num_free:
            self._evict(need - self.num_free, now)
        if need > self.num_free:
            # Roll back the prefix refcounts we took; the caller will preempt.
            for bid in held:
                self._release_one(bid)
            return AllocationResult(False, cached, to_compute, need - self.num_free), []

        for i, h in enumerate(hashes[cached:total_blocks_needed], start=cached):
            bid = self._free.pop()
            blk = self._blocks[bid]
            blk.hash, blk.ref_count, blk.last_used = h, 1, now
            self._by_hash[h] = bid
            held.append(bid)
        for _ in range(need - to_compute):
            bid = self._free.pop()
            blk = self._blocks[bid]
            blk.hash, blk.ref_count, blk.last_used = None, 1, now
            held.append(bid)

        return AllocationResult(True, cached, to_compute, 0), held

    def free(self, block_ids: Sequence[int]) -> None:
        for bid in block_ids:
            self._release_one(bid)

    # -- internals ---------------------------------------------------------
    def _release_one(self, bid: int) -> None:
        blk = self._blocks[bid]
        blk.ref_count -= 1
        assert blk.ref_count >= 0, f"negative refcount on block {bid}"
        if blk.ref_count == 0:
            # Cached blocks stay resident (and reusable) until evicted;
            # uncached blocks return to the pool immediately.
            if blk.hash is None:
                self._free.append(bid)
            else:
                self._free.append(bid)

    def _evict(self, n: int, now: float) -> None:
        """Evict n LRU cached blocks that are free (ref_count == 0)."""
        candidates = [
            bid for bid in self._free
            if self._blocks[bid].hash is not None and self._blocks[bid].ref_count == 0
        ]
        candidates.sort(key=lambda b: self._blocks[b].last_used)
        for bid in candidates[:n]:
            blk = self._blocks[bid]
            if blk.hash in self._by_hash and self._by_hash[blk.hash] == bid:
                del self._by_hash[blk.hash]
            blk.hash = None
            self.evictions += 1
