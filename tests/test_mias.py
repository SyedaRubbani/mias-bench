"""Tests for the allocator's invariants and the metrics' edge cases.

These are the claims the simulator's credibility rests on. If any of them
break, the results in results/ mean nothing.

Run:  python -m pytest -q     (or: python -m unittest discover tests)
"""

from __future__ import annotations

import unittest

from mias.engine import Engine, EngineConfig
from mias.kv import BlockAllocator, block_hash
from mias.metrics import (
    gate_gap_seconds,
    initiative_displacement,
    order_fidelity_tau,
    random_permutation_inversion,
    simultaneity_rate,
    turn_order_inversion,
)
from mias.policies import POLICIES, AgencyPreserving, FCFS
from mias.workload import Turn, WorkloadGenerator


def make_turn(tid, sid, rank, ft, gate=False, rnd=0):
    t = Turn(
        turn_id=tid, session_id=sid, round_idx=rnd, agent=f"a{rank}",
        designed_rank=rank, human_gate_before=gate, token_ids=[1, 2, 3],
        output_tokens=10, arrival_time=0.0,
    )
    t.first_token_time = ft
    return t


class TestBlockHash(unittest.TestCase):
    def test_chaining_is_order_sensitive(self):
        a = block_hash(None, [1, 2, 3])
        b = block_hash(None, [3, 2, 1])
        self.assertNotEqual(a, b)

    def test_parent_changes_child(self):
        p1, p2 = block_hash(None, [1]), block_hash(None, [2])
        self.assertNotEqual(block_hash(p1, [9]), block_hash(p2, [9]))


class TestAllocator(unittest.TestCase):
    def setUp(self):
        self.alloc = BlockAllocator(total_blocks=64, block_size=4)

    def test_conservation(self):
        tokens = list(range(40))
        res, blocks = self.alloc.allocate(tokens, decode_blocks=2, now=0.0)
        self.assertTrue(res.ok)
        self.alloc.free(blocks)
        self.assertEqual(self.alloc.num_free, self.alloc.total_blocks)

    def test_prefix_reuse(self):
        tokens = list(range(40))
        _, b1 = self.alloc.allocate(tokens, 2, 0.0)
        self.alloc.free(b1)
        res, b2 = self.alloc.allocate(tokens, 2, 1.0)
        self.assertGreater(res.cached_blocks, 0, "identical prompt should hit")
        self.assertEqual(res.computed_blocks, 0)
        self.alloc.free(b2)

    def test_divergent_tail_only_shares_prefix(self):
        shared = list(range(32))
        _, b1 = self.alloc.allocate(shared + [900, 901, 902, 903], 1, 0.0)
        res, b2 = self.alloc.allocate(shared + [800, 801, 802, 803], 1, 1.0)
        self.assertEqual(res.cached_blocks, len(shared) // 4)
        self.alloc.free(b1)
        self.alloc.free(b2)

    def test_refcount_never_negative_and_no_double_free(self):
        tokens = list(range(32))
        _, b1 = self.alloc.allocate(tokens, 1, 0.0)
        _, b2 = self.alloc.allocate(tokens, 1, 0.0)   # shares the prefix
        self.alloc.free(b1)
        self.alloc.free(b2)
        for blk in self.alloc._blocks.values():
            self.assertGreaterEqual(blk.ref_count, 0)

    def test_oom_is_reported_not_raised(self):
        big = list(range(4000))
        res, blocks = self.alloc.allocate(big, 4, 0.0)
        self.assertFalse(res.ok)
        self.assertEqual(blocks, [])
        self.assertGreater(res.blocks_short, 0)
        # Failed allocation must leave the pool untouched.
        self.assertEqual(self.alloc.num_free, self.alloc.total_blocks)

    def test_probe_does_not_mutate(self):
        tokens = list(range(32))
        _, b = self.alloc.allocate(tokens, 1, 0.0)
        free_before = self.alloc.num_free
        self.alloc.probe(tokens)
        self.assertEqual(self.alloc.num_free, free_before)
        self.alloc.free(b)


class TestMetrics(unittest.TestCase):
    def test_tau_perfect_order(self):
        turns = [make_turn(i, 0, i, float(i)) for i in range(4)]
        self.assertAlmostEqual(order_fidelity_tau(turns), 1.0)
        self.assertEqual(turn_order_inversion(turns), 0.0)

    def test_tau_reversed_order(self):
        turns = [make_turn(i, 0, i, float(3 - i)) for i in range(4)]
        self.assertAlmostEqual(order_fidelity_tau(turns), -1.0)

    def test_ties_give_zero_tau_and_count_as_failure(self):
        turns = [make_turn(i, 0, i, 5.0) for i in range(4)]
        self.assertAlmostEqual(order_fidelity_tau(turns), 0.0)
        self.assertEqual(turn_order_inversion(turns), 1.0)
        self.assertEqual(simultaneity_rate(turns), 1.0)

    def test_displacement_and_gap(self):
        turns = [
            make_turn(0, 0, 0, 0.0),
            make_turn(1, 0, 1, 0.1, gate=True),   # gate closed in 0.1s
        ]
        self.assertEqual(initiative_displacement(turns, threshold=1.2), 1.0)
        self.assertAlmostEqual(gate_gap_seconds(turns), 0.1)
        turns[1].first_token_time = 2.0
        self.assertEqual(initiative_displacement(turns, threshold=1.2), 0.0)

    def test_random_permutation_baseline(self):
        self.assertAlmostEqual(random_permutation_inversion(4), 1 - 1 / 24)


class TestEngine(unittest.TestCase):
    def _run(self, policy, blocks=2048, rate=0.2, n=6, seed=0):
        gen = WorkloadGenerator(seed=seed)
        sessions = gen.generate(n_sessions=n, arrival_rate=rate)
        engine = Engine(EngineConfig(total_blocks=blocks), policy)
        return engine.run(sessions), engine

    def test_all_turns_complete(self):
        for name, cls in POLICIES.items():
            turns, _ = self._run(cls())
            unfinished = [t for t in turns if t.finish_time is None]
            self.assertEqual(unfinished, [], f"{name} left turns unfinished")

    def test_determinism(self):
        a, _ = self._run(FCFS(), seed=7)
        b, _ = self._run(FCFS(), seed=7)
        self.assertEqual([t.first_token_time for t in a],
                         [t.first_token_time for t in b])

    def test_causality(self):
        turns, _ = self._run(FCFS())
        for t in turns:
            self.assertLessEqual(t.arrival_time, t.first_token_time)
            self.assertLessEqual(t.first_token_time, t.finish_time)

    def test_agency_policy_preserves_designed_order(self):
        turns, _ = self._run(AgencyPreserving(gate_seconds=1.2))
        self.assertAlmostEqual(order_fidelity_tau(turns), 1.0, places=6)
        self.assertEqual(initiative_displacement(turns, 1.2), 0.0)

    def test_fcfs_does_not_preserve_designed_order(self):
        turns, _ = self._run(FCFS())
        self.assertLess(order_fidelity_tau(turns), 0.5)

    def test_preemption_occurs_under_memory_pressure(self):
        turns, _ = self._run(FCFS(), blocks=640, rate=0.4, n=8)
        self.assertGreater(sum(t.preemptions for t in turns), 0)

    def test_pool_is_returned_after_run(self):
        _, engine = self._run(FCFS())
        self.assertEqual(engine.allocator.num_free,
                         engine.allocator.total_blocks)


if __name__ == "__main__":
    unittest.main()
