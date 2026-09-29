#!/usr/bin/env bash
# Reproduce every result and figure in this repository from a clean checkout.
# CPU only, no GPU, no network. Takes about three minutes.
set -euo pipefail
cd "$(dirname "$0")"

echo "== tests =="
python -m unittest discover -s tests

echo; echo "== H1: order fidelity under default scheduling =="
python -m mias.experiments.h1_order_fidelity

echo; echo "== H3: policy trade-off =="
python -m mias.experiments.h3_policy_tradeoff

echo; echo "== figures =="
python -m mias.experiments.make_figures

echo; echo "Done. Results in results/, figures in results/figures/."
