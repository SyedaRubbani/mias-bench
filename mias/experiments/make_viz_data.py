#!/usr/bin/env python3
"""Build results/viz_data.json for the interactive provenance explorer.

The explorer needs per-turn timing for the same rounds under every policy, so
a reader can flip between them and watch the designed order collapse or hold.
That means running identical workloads (same seed, same prompts, same arrival
times) through each policy and keeping the turn-level detail the CSVs
aggregate away.

Run:  python -m mias.experiments.make_viz_data
Out:  results/viz_data.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..engine import Engine, EngineConfig
from ..metrics import (
    gate_gap_seconds,
    initiative_displacement,
    order_fidelity_tau,
    simultaneity_rate,
    throughput,
)
from ..policies import POLICIES
from ..provenance import ProvenanceLog
from ..workload import DEFAULT_AGENTS, WorkloadGenerator

RESULTS = Path("results")
N_SESSIONS = 8
ARRIVAL_RATE = 0.15
GATE = 1.2
CONDITIONS = {
    "kv_large": EngineConfig(total_blocks=4096),
    "kv_tight": EngineConfig(total_blocks=448),
}


def run_one(policy_name: str, cfg: EngineConfig, seed: int = 3):
    gen = WorkloadGenerator(seed=seed)
    sessions = gen.generate(n_sessions=N_SESSIONS, arrival_rate=ARRIVAL_RATE)
    cls = POLICIES[policy_name]
    policy = cls(gate_seconds=GATE) if policy_name == "agency_preserving" else cls()
    log = ProvenanceLog()
    engine = Engine(cfg, policy, log)
    turns = engine.run(sessions)
    return turns, engine, log


def pack_rounds(turns) -> list[dict]:
    groups: dict[tuple[int, int], list] = defaultdict(list)
    for t in turns:
        groups[(t.session_id, t.round_idx)].append(t)

    packed = []
    for (sid, rnd), group in sorted(groups.items()):
        group = sorted(group, key=lambda t: t.designed_rank)
        t0 = min(t.arrival_time for t in group)
        spoke = [t for t in group if t.first_token_time is not None]
        realised = sorted(spoke, key=lambda t: (t.first_token_time, t.designed_rank))
        packed.append({
            "session": sid,
            "round": rnd,
            "t0": round(t0, 4),
            # Realised speaking order by designed rank, ties kept adjacent.
            "realised": [t.designed_rank for t in realised],
            "turns": [{
                "agent": t.agent,
                "rank": t.designed_rank,
                "gate": t.human_gate_before,
                # All times relative to the round's first arrival, so rounds
                # are comparable regardless of when they happened.
                "arrive": round(t.arrival_time - t0, 4),
                "admit": round(t.admit_time - t0, 4) if t.admit_time else None,
                "first": round(t.first_token_time - t0, 4)
                         if t.first_token_time else None,
                "finish": round(t.finish_time - t0, 4) if t.finish_time else None,
                "preempt": t.preemptions,
                "cached": t.cached_blocks,
                "computed": t.computed_blocks,
            } for t in group],
        })
    return packed


def main() -> None:
    data = {
        "agents": [{"name": a.name, "rank": a.designed_rank,
                    "gate": a.human_gate_before} for a in
                   sorted(DEFAULT_AGENTS, key=lambda a: a.designed_rank)],
        "gate_threshold": GATE,
        "conditions": {},
    }

    for cond, cfg in CONDITIONS.items():
        data["conditions"][cond] = {"blocks": cfg.total_blocks, "policies": {}}
        for policy in ("fcfs", "latency_equalised", "agency_preserving"):
            turns, engine, _log = run_one(policy, cfg)
            data["conditions"][cond]["policies"][policy] = {
                "rounds": pack_rounds(turns),
                "summary": {
                    "tau": round(order_fidelity_tau(turns), 4),
                    "simultaneity": round(simultaneity_rate(turns), 4),
                    "displacement": round(initiative_displacement(turns, GATE), 4),
                    "gate_gap": round(gate_gap_seconds(turns), 4),
                    "throughput": round(throughput(turns), 2),
                    "preemptions": sum(t.preemptions for t in turns),
                    "cache_hit": round(engine.allocator.hit_rate(), 4),
                },
            }
            s = data["conditions"][cond]["policies"][policy]["summary"]
            print(f"{cond:<10}{policy:<20} tau={s['tau']:+.2f} "
                  f"simul={s['simultaneity']:.2f} gap={s['gate_gap']:.2f}s "
                  f"thru={s['throughput']:.0f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "viz_data.json"
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"\nwrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
