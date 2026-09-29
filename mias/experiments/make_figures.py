#!/usr/bin/env python3
"""Figures for the paper, regenerated from results/*.csv.

Every figure in the write-up must come from this script and nothing else, so
that a reviewer can reproduce the exact panels from the shipped CSVs without
rerunning the simulations.

Run:  python -m mias.experiments.make_figures
Out:  results/figures/fig1_order_fidelity.png
      results/figures/fig2_policy_tradeoff.png
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("results")
FIGDIR = RESULTS / "figures"

# Categorical palette, fixed order, validated for CVD separation and contrast.
# Marker shape carries identity too, so the encoding is never colour-alone.
PALETTE = {
    "fcfs": "#2563EB",
    "latency_equalised": "#D97706",
    "agency_preserving": "#BE185D",
}
MARKERS = {"fcfs": "o", "latency_equalised": "s", "agency_preserving": "D"}
LABELS = {
    "fcfs": "FCFS (vLLM default)",
    "latency_equalised": "Latency-equalised",
    "agency_preserving": "Agency-preserving",
}
INK = "#1f2328"
MUTED = "#6b7280"


def load(path: Path) -> List[Dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    ax.set_axisbelow(True)


def fig1() -> Path:
    rows = load(RESULTS / "h1_order_fidelity.csv")
    configs = ["kv_large", "kv_medium", "kv_tight"]
    rates = sorted({float(r["arrival_rate"]) for r in rows})

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    _style(ax)
    shades = {"kv_large": "#93c5fd", "kv_medium": "#3b82f6", "kv_tight": "#1e3a8a"}
    for cfg in configs:
        xs, ys, errs = [], [], []
        for rate in rates:
            sub = [r for r in rows if r["config"] == cfg
                   and float(r["arrival_rate"]) == rate
                   and float(r["dispatch_jitter"]) > 0]
            if not sub:
                continue
            vals = [float(r["order_tau"]) for r in sub]
            xs.append(rate)
            ys.append(mean(vals))
            errs.append(stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0)
        ax.errorbar(xs, ys, yerr=errs, marker="o", markersize=6, linewidth=2,
                    color=shades[cfg], capsize=3, label=cfg.replace("_", " "))
        ax.annotate(cfg.replace("kv_", "KV "), (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(8, -2),
                    fontsize=9, color=shades[cfg], va="center")

    ax.axhline(0.0, color=MUTED, linewidth=1.2, linestyle="--")
    ax.annotate("τ = 0: realised order carries no information about the design",
                (rates[0], 0.02), fontsize=8.5, color=MUTED, va="bottom")
    ax.set_ylim(-0.15, 1.02)
    ax.set_xlabel("Session arrival rate (sessions / s)", fontsize=10, color=INK)
    ax.set_ylabel("Order fidelity τ\n(designed rank vs realised order)",
                  fontsize=10, color=INK)
    ax.set_title("H1 · Under default scheduling the interaction design does not "
                 "survive the serving layer",
                 fontsize=11.5, color=INK, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "fig1_order_fidelity.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig2() -> Path:
    rows = load(RESULTS / "h3_policy_tradeoff.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=160)

    # Panel A: order fidelity by policy and load.
    _style(ax1)
    rates = sorted({float(r["arrival_rate"]) for r in rows})
    width = 0.24
    for i, pol in enumerate(PALETTE):
        ys = []
        for rate in rates:
            sub = [r for r in rows if r["policy"] == pol
                   and float(r["arrival_rate"]) == rate]
            ys.append(mean(float(r["order_tau"]) for r in sub))
        xs = [j + (i - 1) * (width + 0.02) for j in range(len(rates))]
        ax1.bar(xs, ys, width=width, color=PALETTE[pol], label=LABELS[pol],
                edgecolor="white", linewidth=2)
        for x, y in zip(xs, ys):
            ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                         xytext=(0, 4), ha="center", fontsize=8, color=INK)
    ax1.set_xticks(range(len(rates)))
    ax1.set_xticklabels([f"{r:g}" for r in rates])
    ax1.set_ylim(0, 1.15)
    ax1.set_xlabel("Session arrival rate (sessions / s)", fontsize=10, color=INK)
    ax1.set_ylabel("Order fidelity τ", fontsize=10, color=INK)
    ax1.set_title("A · Designed turn order restored", fontsize=11, color=INK,
                  loc="left", pad=10)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper center",
               bbox_to_anchor=(0.5, -0.16), ncol=3, labelcolor=INK,
               handlelength=1.2, columnspacing=1.2)

    # Panel B: what it costs — human floor time against throughput.
    _style(ax2)
    label_offsets = {
        "fcfs": (-6, 14),
        "latency_equalised": (-6, -18),
        "agency_preserving": (-6, 12),
    }
    for pol in PALETTE:
        xs, ys = [], []
        for rate in rates:
            sub = [r for r in rows if r["policy"] == pol
                   and float(r["arrival_rate"]) == rate]
            xs.append(mean(float(r["throughput_tok_s"]) for r in sub))
            ys.append(mean(float(r["gate_gap_s"]) for r in sub))
        ax2.plot(xs, ys, marker=MARKERS[pol], markersize=8, linewidth=2,
                 color=PALETTE[pol], label=LABELS[pol],
                 markeredgecolor="white", markeredgewidth=1.5)
        ax2.annotate(LABELS[pol], (xs[-1], ys[-1]), textcoords="offset points",
                     xytext=label_offsets[pol], fontsize=8.5,
                     color=PALETTE[pol], ha="right")
        x_right = xs[-1]
    ax2.axhline(1.2, color=MUTED, linewidth=1.2, linestyle="--")
    ax2.annotate("human reaction threshold (1.2 s)", (x_right, 1.14),
                 fontsize=8.5, color=MUTED, va="top", ha="right")
    ax2.set_ylim(-0.35, 1.9)
    ax2.set_xlabel("Throughput (output tokens / s)", fontsize=10, color=INK)
    ax2.set_ylabel("Human floor time at designed gates (s)", fontsize=10, color=INK)
    ax2.set_title("B · At a throughput cost of 1–4%", fontsize=11, color=INK,
                  loc="left", pad=10)

    fig.suptitle("H3 · Interaction-aware scheduling restores turn-taking at "
                 "bounded cost", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "fig2_policy_tradeoff.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    for fn in (fig1, fig2):
        print(f"wrote {fn()}")


if __name__ == "__main__":
    main()
