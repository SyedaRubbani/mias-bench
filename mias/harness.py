"""Shared experiment harness: sweep seeds x load x policy, write tidy CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .engine import Engine, EngineConfig
from .metrics import summarise
from .policies import POLICIES
from .provenance import ProvenanceLog
from .workload import WorkloadGenerator

RESULTS = Path("results")


def run_condition(
    policy_name: str,
    seed: int,
    arrival_rate: float,
    n_sessions: int,
    cfg: EngineConfig,
    gate_threshold: float = 1.2,
    log_path: Path | None = None,
    dispatch_jitter: float = 0.005,
) -> Dict[str, float]:
    gen = WorkloadGenerator(seed=seed, dispatch_jitter=dispatch_jitter)
    sessions = gen.generate(n_sessions=n_sessions, arrival_rate=arrival_rate)
    policy_cls = POLICIES[policy_name]
    policy = (
        policy_cls(gate_seconds=gate_threshold)
        if policy_name == "agency_preserving"
        else policy_cls()
    )
    log = ProvenanceLog(enabled=log_path is not None)
    engine = Engine(cfg, policy, log)
    turns = engine.run(sessions)
    if log_path is not None:
        log.write(log_path)
    # Wall-clock timing is deliberately NOT recorded: every column in the CSV
    # must be deterministic so reruns are byte-identical.
    row = {
        "policy": policy_name,
        "seed": seed,
        "arrival_rate": arrival_rate,
        "dispatch_jitter": dispatch_jitter,
        "total_blocks": cfg.total_blocks,
        "gate_threshold": gate_threshold,
        "engine_steps": engine.steps,
        "allocator_hit_rate": round(engine.allocator.hit_rate(), 4),
        "evictions": engine.allocator.evictions,
    }
    row.update(summarise(turns, gate_threshold))
    return row


def sweep(
    policies: Sequence[str],
    seeds: Iterable[int],
    arrival_rates: Sequence[float],
    n_sessions: int,
    cfgs: Dict[str, EngineConfig],
    gate_threshold: float = 1.2,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for cfg_name, cfg in cfgs.items():
        for rate in arrival_rates:
            for policy in policies:
                for seed in seeds:
                    row = run_condition(
                        policy, seed, rate, n_sessions, cfg, gate_threshold
                    )
                    row["config"] = cfg_name
                    rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, float]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path
