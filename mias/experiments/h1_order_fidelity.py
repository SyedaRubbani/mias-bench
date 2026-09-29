#!/usr/bin/env python3
"""H1 - the serving layer overrides the interaction design.

H1  Under vLLM's default scheduling (FCFS admission, LIFO preemption), the
    realised first-token order of a round of agents carries no information
    about the designed turn order: order fidelity (Kendall's tau between
    designed rank and realised order) is indistinguishable from zero, and the
    human's intervention window at designed gates closes far faster than human
    reaction time. This holds with agent content held constant.

What would falsify it: tau reliably above ~0.5 with a bootstrap interval
excluding zero, and gate gaps at or above the reaction threshold. Then the
default scheduler is effectively order-preserving and the project's premise
collapses.

Controls in this experiment:
  * dispatch_jitter 0.005 (async fan-out, realistic) vs 0.0 (ordered dispatch)
    - isolates how much of the effect is arrival order rather than engine
      mechanics.
  * three KV pool sizes - isolates the contribution of memory pressure and
    preemption.

Run:  python -m mias.experiments.h1_order_fidelity
Out:  results/h1_order_fidelity.csv
      results/h1_provenance_sample.jsonl
"""

from __future__ import annotations

import statistics

from ..engine import EngineConfig
from ..harness import RESULTS, run_condition, write_csv
from ..metrics import random_permutation_inversion

SEEDS = range(12)
ARRIVAL_RATES = [0.05, 0.15, 0.30]
JITTERS = [0.005, 0.0]
KV_CONFIGS = {
    "kv_large": EngineConfig(total_blocks=4096),
    "kv_medium": EngineConfig(total_blocks=1536),
    "kv_tight": EngineConfig(total_blocks=768),
}
N_SESSIONS = 16


def main() -> None:
    rows = []
    for cfg_name, cfg in KV_CONFIGS.items():
        for rate in ARRIVAL_RATES:
            for jitter in JITTERS:
                for seed in SEEDS:
                    row = run_condition(
                        "fcfs", seed, rate, N_SESSIONS, cfg,
                        dispatch_jitter=jitter,
                    )
                    row["config"] = cfg_name
                    rows.append(row)
    path = write_csv(rows, RESULTS / "h1_order_fidelity.csv")

    run_condition(
        "fcfs", seed=0, arrival_rate=0.15, n_sessions=6,
        cfg=KV_CONFIGS["kv_medium"],
        log_path=RESULTS / "h1_provenance_sample.jsonl",
    )

    print(f"wrote {path}")
    print(f"random-permutation inversion baseline for 4 agents: "
          f"{random_permutation_inversion(4):.3f}\n")
    hdr = (f"{'config':<10}{'rate':>6}{'jitter':>8}{'tau':>7}{'tau_lo':>8}"
           f"{'tau_hi':>8}{'inv':>7}{'simul':>7}{'gap_s':>7}{'preempt':>9}")
    print(hdr)
    for cfg_name in KV_CONFIGS:
        for rate in ARRIVAL_RATES:
            for jitter in JITTERS:
                sub = [r for r in rows if r["config"] == cfg_name
                       and r["arrival_rate"] == rate
                       and r["dispatch_jitter"] == jitter]
                m = lambda k: statistics.mean(r[k] for r in sub)  # noqa: E731
                print(f"{cfg_name:<10}{rate:>6.2f}{jitter:>8.3f}"
                      f"{m('order_tau'):>7.2f}{m('order_tau_ci_lo'):>8.2f}"
                      f"{m('order_tau_ci_hi'):>8.2f}{m('inversion_rate'):>7.2f}"
                      f"{m('simultaneity_rate'):>7.2f}{m('gate_gap_s'):>7.2f}"
                      f"{m('preemptions_per_turn'):>9.2f}")


if __name__ == "__main__":
    main()
