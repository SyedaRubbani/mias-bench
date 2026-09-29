"""Outcome measures.

Two are standard serving metrics (TTFT distribution, throughput) and two are
the interaction measures this project introduces:

  turn-order inversion rate
      the fraction of interaction rounds in which the realised first-token
      order differs from the designed turn order. An inversion means the
      serving layer overrode the interaction design.

  initiative displacement rate
      the fraction of *human gates* - points where the design intends the
      human to be able to intervene - that closed faster than the human
      reaction threshold. This is the measure that should track perceived
      agency in the user study; here it is a purely mechanical proxy.

Uncertainty is reported as a bootstrap percentile interval over sessions,
because sessions, not turns, are the independent unit.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .workload import Turn


def _rounds(turns: Sequence[Turn]) -> Dict[Tuple[int, int], List[Turn]]:
    groups: Dict[Tuple[int, int], List[Turn]] = defaultdict(list)
    for t in turns:
        groups[(t.session_id, t.round_idx)].append(t)
    return groups


def turn_order_inversion(turns: Sequence[Turn]) -> float:
    """Fraction of rounds in which the designed order was NOT realised.

    A tie counts as a failure. Under batched prefill several agents in a round
    routinely finish prefill in the same engine step and therefore emit their
    first token simultaneously; the design is then not reordered but
    flattened, which is at least as damaging to turn-taking. See
    `simultaneity_rate`.
    """
    failed = total = 0
    for group in _rounds(turns).values():
        spoke = [t for t in group if t.first_token_time is not None]
        if len(spoke) < 2:
            continue
        total += 1
        ordered = sorted(spoke, key=lambda x: (x.first_token_time, x.designed_rank))
        ranks = [t.designed_rank for t in ordered]
        times = [t.first_token_time for t in ordered]
        tied = any(a == b for a, b in zip(times, times[1:]))
        if tied or ranks != sorted(ranks):
            failed += 1
    return failed / total if total else 0.0


def simultaneity_rate(turns: Sequence[Turn]) -> float:
    """Fraction of rounds where two or more agents share a first-token instant."""
    simultaneous = total = 0
    for group in _rounds(turns).values():
        times = [t.first_token_time for t in group if t.first_token_time is not None]
        if len(times) < 2:
            continue
        total += 1
        if len(set(times)) < len(times):
            simultaneous += 1
    return simultaneous / total if total else 0.0


def initiative_displacement(turns: Sequence[Turn], threshold: float = 1.2) -> float:
    """Fraction of human gates that closed before the human could act."""
    displaced = total = 0
    for group in _rounds(turns).values():
        by_rank = {t.designed_rank: t for t in group}
        for t in group:
            if not t.human_gate_before or t.first_token_time is None:
                continue
            pred = by_rank.get(t.designed_rank - 1)
            if pred is None or pred.first_token_time is None:
                continue
            total += 1
            if t.first_token_time - pred.first_token_time < threshold:
                displaced += 1
    return displaced / total if total else 0.0


def order_fidelity_tau(turns: Sequence[Turn]) -> float:
    """Kendall's tau between designed turn order and realised first-token order.

    tau =  1  the serving layer reproduced the interaction design exactly
    tau =  0  realised order is indistinguishable from a random permutation
              of the designed order - i.e. the design had no effect
    tau = -1  the design was exactly inverted

    This is the primary dependent variable. The binary `turn_order_inversion`
    saturates at 1 - 1/k! for k agents once ordering is arbitrary, so it
    cannot show *how much* order was lost; tau can.
    """
    taus: List[float] = []
    for group in _rounds(turns).values():
        spoke = [t for t in group if t.first_token_time is not None]
        k = len(spoke)
        if k < 2:
            continue
        concordant = discordant = 0
        for i in range(k):
            for j in range(i + 1, k):
                a, b = spoke[i], spoke[j]
                d = (a.designed_rank - b.designed_rank)
                r = (a.first_token_time - b.first_token_time)
                if d == 0 or r == 0:
                    continue
                if (d < 0) == (r < 0):
                    concordant += 1
                else:
                    discordant += 1
        n_pairs = k * (k - 1) / 2
        if n_pairs:
            taus.append((concordant - discordant) / n_pairs)
    return sum(taus) / len(taus) if taus else 0.0


def random_permutation_inversion(k: int) -> float:
    """Inversion rate expected if realised order were a uniform permutation."""
    fact = 1
    for i in range(2, k + 1):
        fact *= i
    return 1.0 - 1.0 / fact


def gate_gap_seconds(turns: Sequence[Turn]) -> float:
    """Mean seconds of floor the human actually gets at each designed gate.

    The continuous counterpart of `initiative_displacement`: displacement is
    this quantity thresholded, and thresholding throws away the effect size.
    Report both.
    """
    gaps: List[float] = []
    for group in _rounds(turns).values():
        by_rank = {t.designed_rank: t for t in group}
        for t in group:
            if not t.human_gate_before or t.first_token_time is None:
                continue
            pred = by_rank.get(t.designed_rank - 1)
            if pred is None or pred.first_token_time is None:
                continue
            gaps.append(t.first_token_time - pred.first_token_time)
    return sum(gaps) / len(gaps) if gaps else 0.0


def ttft_stats(turns: Sequence[Turn]) -> Dict[str, float]:
    vals = sorted(t.ttft for t in turns if t.ttft is not None)
    if not vals:
        return {}
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0

    def pct(p: float) -> float:
        k = min(n - 1, max(0, int(round(p * (n - 1)))))
        return vals[k]

    return {
        "ttft_mean": mean,
        "ttft_sd": math.sqrt(var),
        "ttft_cv": math.sqrt(var) / mean if mean else 0.0,
        "ttft_p50": pct(0.50),
        "ttft_p95": pct(0.95),
        "ttft_iqr": pct(0.75) - pct(0.25),
    }


def throughput(turns: Sequence[Turn]) -> float:
    finished = [t for t in turns if t.finish_time is not None]
    if not finished:
        return 0.0
    span = max(t.finish_time for t in finished) - min(t.arrival_time for t in finished)
    tokens = sum(t.output_tokens for t in finished)
    return tokens / span if span > 0 else 0.0


def preemption_rate(turns: Sequence[Turn]) -> float:
    return sum(t.preemptions for t in turns) / len(turns) if turns else 0.0


def cache_hit_rate(turns: Sequence[Turn]) -> float:
    c = sum(t.cached_blocks for t in turns)
    m = sum(t.computed_blocks for t in turns)
    return c / (c + m) if (c + m) else 0.0


def bootstrap_ci(
    turns: Sequence[Turn],
    statistic: Callable[[Sequence[Turn]], float],
    n_boot: int = 600,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile bootstrap over sessions (the independent unit)."""
    by_session: Dict[int, List[Turn]] = defaultdict(list)
    for t in turns:
        by_session[t.session_id].append(t)
    sessions = list(by_session.values())
    point = statistic(turns)
    if len(sessions) < 2:
        return point, point, point
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sample: List[Turn] = []
        for _ in range(len(sessions)):
            sample.extend(rng.choice(sessions))
        draws.append(statistic(sample))
    draws.sort()
    lo = draws[int(alpha / 2 * (n_boot - 1))]
    hi = draws[int((1 - alpha / 2) * (n_boot - 1))]
    return point, lo, hi


def summarise(turns: Sequence[Turn], gate_threshold: float = 1.2) -> Dict[str, float]:
    inv, inv_lo, inv_hi = bootstrap_ci(turns, turn_order_inversion)
    tau, tau_lo, tau_hi = bootstrap_ci(turns, order_fidelity_tau)
    dis, dis_lo, dis_hi = bootstrap_ci(
        turns, lambda ts: initiative_displacement(ts, gate_threshold)
    )
    out = {
        "order_tau": tau,
        "order_tau_ci_lo": tau_lo,
        "order_tau_ci_hi": tau_hi,
        "inversion_rate": inv,
        "inversion_ci_lo": inv_lo,
        "inversion_ci_hi": inv_hi,
        "displacement_rate": dis,
        "displacement_ci_lo": dis_lo,
        "displacement_ci_hi": dis_hi,
        "gate_gap_s": gate_gap_seconds(turns),
        "simultaneity_rate": simultaneity_rate(turns),
        "throughput_tok_s": throughput(turns),
        "preemptions_per_turn": preemption_rate(turns),
        "cache_hit_rate": cache_hit_rate(turns),
        "n_turns": float(len(turns)),
    }
    out.update(ttft_stats(turns))
    return out
