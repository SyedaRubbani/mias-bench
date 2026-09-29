#!/usr/bin/env python3
"""H3 - interaction-aware scheduling restores turn-taking at bounded cost.

H3  A scheduler given the designed turn order and the human reaction threshold
    restores order fidelity (tau -> 1) and reopens the human's intervention
    window, at a throughput cost small enough to be worth paying.

"Small enough" is a judgement, so this experiment reports the trade-off curve
rather than a single verdict: order fidelity, gate gap and TTFT against
throughput, for all three policies across load and KV pressure.

Note what this experiment does NOT establish. It shows the *mechanism* can be
controlled. Whether restoring the mechanism changes human agency, trust
calibration or intervention behaviour is H2, and H2 requires a user study.
Nothing in this repository tests H2. See README.md, "Scope and honest limits".

Run:  python -m mias.experiments.h3_policy_tradeoff
Out:  results/h3_policy_tradeoff.csv
"""

from __future__ import annotations

import statistics

from ..engine import EngineConfig
from ..harness import RESULTS, run_condition, write_csv

SEEDS = range(12)
ARRIVAL_RATES = [0.05, 0.15, 0.30]
POLICIES = ["fcfs", "latency_equalised", "agency_preserving"]
KV_CONFIGS = {
    "kv_large": EngineConfig(total_blocks=4096),
    "kv_tight": EngineConfig(total_blocks=1024),
}
N_SESSIONS = 16
GATE_THRESHOLD = 1.2


def main() -> None:
    rows = []
    for cfg_name, cfg in KV_CONFIGS.items():
        for rate in ARRIVAL_RATES:
            for policy in POLICIES:
                for seed in SEEDS:
                    row = run_condition(
                        policy, seed, rate, N_SESSIONS, cfg,
                        gate_threshold=GATE_THRESHOLD,
                    )
                    row["config"] = cfg_name
                    rows.append(row)
    path = write_csv(rows, RESULTS / "h3_policy_tradeoff.csv")
    print(f"wrote {path}\n")

    hdr = (f"{'config':<10}{'rate':>6}  {'policy':<18}{'tau':>6}{'gap_s':>7}"
           f"{'disp':>7}{'ttft':>7}{'thru':>9}{'thru_%':>8}")
    print(hdr)
    for cfg_name in KV_CONFIGS:
        for rate in ARRIVAL_RATES:
            base = statistics.mean(
                r["throughput_tok_s"] for r in rows
                if r["config"] == cfg_name and r["arrival_rate"] == rate
                and r["policy"] == "fcfs"
            )
            for policy in POLICIES:
                sub = [r for r in rows if r["config"] == cfg_name
                       and r["arrival_rate"] == rate and r["policy"] == policy]
                m = lambda k: statistics.mean(r[k] for r in sub)  # noqa: E731
                thru = m("throughput_tok_s")
                print(f"{cfg_name:<10}{rate:>6.2f}  {policy:<18}"
                      f"{m('order_tau'):>6.2f}{m('gate_gap_s'):>7.2f}"
                      f"{m('displacement_rate'):>7.2f}{m('ttft_mean'):>7.2f}"
                      f"{thru:>9.1f}{100 * (thru - base) / base:>7.1f}%")


if __name__ == "__main__":
    main()
