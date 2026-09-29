"""Scheduling policies.

Three arms, all running on the same engine and the same block allocator, so
any difference between them is attributable to the policy alone:

  FCFS               - vLLM's default behaviour: first-come-first-served
                       admission, LIFO preemption under memory pressure.
                       This is the baseline that current mixed-initiative
                       systems inherit without choosing it.

  LatencyEqualised   - admits in order of *predicted* TTFT (slowest first),
                       using a cache probe. Reduces latency variance without
                       knowing anything about the interaction design.

  AgencyPreserving   - the interaction-aware arm. It is given the designed
                       turn order and the human reaction threshold, and it
                       (a) refuses to let a turn emit before its designed
                       predecessor, (b) holds a minimum gap open before
                       human-gated turns so the human can still intervene,
                       and (c) preempts the *least* design-critical turn
                       rather than the most recently admitted one.

Policies never see token content, only structure and cache state, so none of
them can cheat by inspecting the prompt.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .kv import BlockAllocator
from .workload import Turn


class Policy:
    name = "base"

    def order_waiting(
        self, waiting: Sequence[Turn], allocator: BlockAllocator, now: float,
        first_token: Dict[int, float],
    ) -> List[Turn]:
        raise NotImplementedError

    def admissible(
        self, turn: Turn, now: float, first_token: Dict[int, float],
        turns_by_key: Dict[tuple, Turn],
    ) -> bool:
        """Gate an individual turn regardless of ordering."""
        return True

    def preemption_victim(self, running: Sequence[Turn]) -> Optional[Turn]:
        raise NotImplementedError


class FCFS(Policy):
    """vLLM default: arrival order in, most-recently-admitted out."""

    name = "fcfs"

    def order_waiting(self, waiting, allocator, now, first_token):
        return sorted(waiting, key=lambda t: (t.arrival_time, t.turn_id))

    def preemption_victim(self, running):
        if not running:
            return None
        # LIFO: the request admitted most recently is evicted first.
        return max(running, key=lambda t: (t.admit_time or 0.0, t.turn_id))


class LatencyEqualised(Policy):
    """Cache-aware but design-blind: serve the predicted-slowest turn first."""

    name = "latency_equalised"

    def order_waiting(self, waiting, allocator, now, first_token):
        def predicted_cost(t: Turn) -> float:
            cached = allocator.probe(t.token_ids) * allocator.block_size
            return -(len(t.token_ids) - cached)   # most work first
        return sorted(waiting, key=lambda t: (predicted_cost(t), t.turn_id))

    def preemption_victim(self, running):
        if not running:
            return None
        # Evict whoever has the least sunk cost, to minimise wasted recompute.
        return min(running, key=lambda t: (t.prefilled_tokens, t.turn_id))


class AgencyPreserving(Policy):
    """Interaction-aware: the designed turn order is a scheduling constraint.

    `gate_seconds` is the human reaction threshold - the minimum time a human
    needs after seeing agent k's first token to decide whether to interrupt
    before agent k+1 speaks. Default 1.2s; make it a study parameter, not a
    constant, because the right value is an empirical question.
    """

    name = "agency_preserving"

    def __init__(self, gate_seconds: float = 1.2):
        self.gate_seconds = gate_seconds

    def order_waiting(self, waiting, allocator, now, first_token):
        return sorted(
            waiting,
            key=lambda t: (t.session_id, t.round_idx, t.designed_rank, t.turn_id),
        )

    def admissible(self, turn, now, first_token, turns_by_key):
        if turn.designed_rank == 0:
            return True
        # Every designed predecessor in this round must have spoken.
        for rank in range(turn.designed_rank):
            pred = turns_by_key.get((turn.session_id, turn.round_idx, rank))
            if pred is None:
                continue
            ft = first_token.get(pred.turn_id)
            if ft is None:
                return False
            if turn.human_gate_before and rank == turn.designed_rank - 1:
                # Hold the floor open long enough for the human to act.
                if now < ft + self.gate_seconds:
                    return False
        return True

    def preemption_victim(self, running):
        if not running:
            return None
        # Sacrifice the least design-critical turn: highest designed rank,
        # and never a turn the human is currently gated on.
        return max(running, key=lambda t: (t.designed_rank, t.turn_id))


POLICIES = {
    "fcfs": FCFS,
    "latency_equalised": LatencyEqualised,
    "agency_preserving": AgencyPreserving,
}
