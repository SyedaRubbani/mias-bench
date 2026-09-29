#!/usr/bin/env python3
"""Build notebooks/colab_t4_measurement.ipynb.

The notebook is generated rather than hand-written so its JSON is always
valid and its code cells can be linted as ordinary Python before shipping.

Run:  python tools/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("notebooks/colab_t4_measurement.ipynb")

MD_INTRO = """\
# Measuring the scheduler on a real GPU (Colab T4)

This notebook replaces the simulator's cost model with **measurements from a
real vLLM server**. It runs end to end on a free Colab **T4**.

What it produces:

1. **Calibration** — fits the simulator's step-cost constants to this
   deployment, so simulated numbers stop being placeholders.
2. **H1 on hardware** — fires a round of four agents concurrently, the way a
   real multi-agent framework does, and measures whether the realised
   first-token order carries any information about the designed turn order
   (Kendall's τ).
3. **H3 on hardware** — reruns the same rounds with vLLM's `priority`
   scheduling and with client-side gating, and measures what restoring the
   order costs in throughput.

Everything is scored by `mias.metrics`, the same code that scores the
simulator, so the measured and simulated results sit in the same table.

---

### What a T4 can and cannot tell you

**Can.** The mechanism is real at this scale. Prefix caching, continuous
batching, chunked prefill and preemption all operate on a T4 exactly as they
do on an A100 — you can make the KV pool as tight as you like with
`--num-gpu-blocks-override`, which is the lever that matters for this study.

**Cannot.**

- **16 GB, no bfloat16.** T4 is Turing (SM 7.5), so the server must run
  `--dtype half`, and FlashAttention needs SM 8.0+, so vLLM falls back to
  another attention backend. Absolute latencies are not comparable to an
  A100 deployment; the *ordering* effects are what transfer.
- **Small model.** A 1.5B model in fp16 leaves plenty of KV headroom, which
  is good for control but means per-request prefill is cheap. Expect smaller
  absolute gaps than a 7B+ deployment would show.
- **Shared, unpinnable hardware.** Colab does not permit locking GPU clocks,
  and the VM is virtualised. Report medians and interquartile ranges over
  many repeats, never a single mean, and never claim precision you cannot
  defend. This is stated as a limitation in the paper, not hidden.

If you later get an A100 or L4, nothing in the notebook changes except the
runtime you select and the model you pass.
"""

MD_SETUP = """\
## 1 · Environment

Set **Runtime → Change runtime type → T4 GPU** before running anything.
The vLLM install pulls its own PyTorch build and takes several minutes.
"""

CODE_GPU = """\
!nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
"""

CODE_INSTALL = """\
# vLLM brings its own torch build; this takes 5-10 minutes on Colab.
!pip install -q vllm aiohttp
!git clone -q https://github.com/REPLACE-ME/mias-bench.git 2>/dev/null || true
%cd /content/mias-bench
!python -m unittest discover -s tests 2>&1 | tail -3
"""

MD_LAUNCH = """\
## 2 · Launch vLLM

`launch()` starts a server and waits for `/health`. The two arguments that
matter for this study:

- `num_gpu_blocks_override` — sets the KV pool size directly. This is how the
  `kv_large` / `kv_tight` conditions from the simulator are reproduced on real
  hardware. Leave it `None` for the default (abundant) pool.
- `priority` — enables vLLM's priority scheduling, needed for the `priority`
  dispatch arm.

T4-specific flags are set for you: `--dtype half` (Turing has no bfloat16)
and a conservative `--gpu-memory-utilization`.
"""

CODE_LAUNCH = '''\
import os, signal, subprocess, time, urllib.request

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # ~3.1 GB in fp16; fits a T4 with room
BASE_URL = "http://127.0.0.1:8000"
_server = None


def launch(num_gpu_blocks_override=None, priority=False,
           max_num_seqs=16, max_model_len=4096, timeout=900):
    """Start vLLM in the background and block until it is serving."""
    global _server
    shutdown()
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--dtype", "half",                 # T4 is Turing: no bfloat16
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--gpu-memory-utilization", "0.85",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--disable-log-requests",
        "--port", "8000",
    ]
    if num_gpu_blocks_override:
        cmd += ["--num-gpu-blocks-override", str(num_gpu_blocks_override)]
    if priority:
        cmd += ["--scheduling-policy", "priority"]

    log = open("/content/vllm.log", "w")
    _server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                               preexec_fn=os.setsid)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _server.poll() is not None:
            print(open("/content/vllm.log").read()[-4000:])
            raise RuntimeError("vLLM exited during startup - see log above")
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2) as r:
                if r.status == 200:
                    print(f"server up in {time.time() - t0:.0f}s  "
                          f"(blocks_override={num_gpu_blocks_override}, "
                          f"priority={priority})")
                    return
        except Exception:
            time.sleep(3)
    raise TimeoutError("vLLM did not become healthy; check /content/vllm.log")


def shutdown():
    """Stop the server and free the GPU before relaunching."""
    global _server
    if _server is not None and _server.poll() is None:
        os.killpg(os.getpgid(_server.pid), signal.SIGTERM)
        _server.wait(timeout=120)
    _server = None
    time.sleep(5)


def kv_blocks_reported():
    """How many KV blocks the server actually allocated, from its own log."""
    import re
    text = open("/content/vllm.log").read()
    m = re.findall(r"GPU (?:KV cache size|blocks)[:=] *([\\d,]+)", text)
    return m[-1] if m else "unknown"
'''

MD_CAL = """\
## 3 · Calibration

Sends isolated, cache-cold requests of increasing length and regresses TTFT on
prompt length. **Check the R².** Below about 0.9, the simulator's linear step
model does not describe this deployment, and the honest move is to report that
rather than to use the fitted constants anyway.
"""

CODE_CAL = '''\
import asyncio, json
from mias.measure import calibrate

launch()
fit = await calibrate(BASE_URL, MODEL, prompt_words=(200, 600, 1200, 2400),
                      repeats=6)
print(json.dumps(fit, indent=2))
if fit["r2"] < 0.9:
    print("\\nWARNING: R2 < 0.9 - the linear step model does not describe "
          "this deployment. Report this; do not quietly use the constants.")
open("results/calibration_t4.json", "w").write(json.dumps(fit, indent=2))
'''

MD_MEASURE = """\
## 4 · H1 and H3 on hardware

Two KV-pool conditions × three dispatch modes. Each condition needs its own
server, so this cell relaunches between conditions and takes roughly
20–35 minutes in total on a T4.

- `concurrent` — the baseline: everything fired at once, server decides.
- `priority` — same, but each request carries its designed rank as vLLM's
  `priority`. An in-engine lever that needs no fork.
- `gated` — client-side gating on the designed order. Guarantees turn-taking
  and pays for it in lost overlap; the gap between this and `priority` is the
  argument for doing it inside the scheduler.
"""

CODE_MEASURE = '''\
import csv
from mias.measure import build_rounds, run_measurement
from mias.metrics import summarise

N_SESSIONS = 10
ROUNDS_PER_SESSION = 2
GATE_SECONDS = 1.2
REPEATS = 3          # independent repeats per condition, for medians

CONDITIONS = [
    ("kv_large", None),
    ("kv_tight", 512),     # tighten until the log reports preemptions
]
MODES = ["concurrent", "priority", "gated"]

rows = []
for cond_name, blocks in CONDITIONS:
    launch(num_gpu_blocks_override=blocks, priority=True)
    print(f"{cond_name}: KV blocks reported = {kv_blocks_reported()}")
    for mode in MODES:
        for rep in range(REPEATS):
            rounds = build_rounds(
                n_sessions=N_SESSIONS,
                rounds_per_session=ROUNDS_PER_SESSION,
                max_tokens=96,
                seed=1000 * rep,        # a fresh corpus defeats stale caching
            )
            turns, delta = await run_measurement(
                BASE_URL, MODEL, rounds, mode=mode,
                gate_seconds=GATE_SECONDS, warmup_rounds=2,
            )
            row = {"source": "measured_t4", "config": cond_name,
                   "mode": mode, "repeat": rep}
            row.update(summarise(turns, GATE_SECONDS))
            row.update({k.replace("vllm:", ""): v for k, v in delta.items()})
            rows.append(row)
            print(f"  {cond_name:<9} {mode:<11} rep{rep}  "
                  f"tau={row['order_tau']:+.2f}  "
                  f"gap={row['gate_gap_s']:.2f}s  "
                  f"ttft={row['ttft_mean']:.2f}s  "
                  f"thru={row['throughput_tok_s']:.0f} tok/s")
shutdown()

fields = sorted({k for r in rows for k in r})
with open("results/measured_t4.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print("\\nwrote results/measured_t4.csv")
'''

MD_COMPARE = """\
## 5 · Measured against simulated

The claim worth making to a reviewer is not "the simulator was right" but
"here is where it was right and where it was not". Report both columns and
discuss the gap.
"""

CODE_COMPARE = '''\
import csv, statistics as st

measured = list(csv.DictReader(open("results/measured_t4.csv")))
simulated = list(csv.DictReader(open("results/h3_policy_tradeoff.csv")))

SIM_FOR_MODE = {"concurrent": "fcfs", "priority": "fcfs",
                "gated": "agency_preserving"}

def med(rows, key):
    vals = [float(r[key]) for r in rows]
    return st.median(vals) if vals else float("nan")

print(f"{'mode':<12}{'source':<12}{'tau':>7}{'gap_s':>8}{'ttft_s':>8}{'tok/s':>9}")
for mode in ["concurrent", "priority", "gated"]:
    m = [r for r in measured if r["mode"] == mode]
    s = [r for r in simulated if r["policy"] == SIM_FOR_MODE[mode]]
    for label, rows_ in (("measured T4", m), ("simulated", s)):
        if not rows_:
            continue
        print(f"{mode:<12}{label:<12}{med(rows_, 'order_tau'):>7.2f}"
              f"{med(rows_, 'gate_gap_s'):>8.2f}"
              f"{med(rows_, 'ttft_mean'):>8.2f}"
              f"{med(rows_, 'throughput_tok_s'):>9.0f}")
    print()

print("Reminder: the simulator has no `priority` arm - vLLM's priority "
      "scheduling has no counterpart in the reference model yet. That row "
      "is measurement only, and it is the most interesting one: it shows "
      "how much order an existing in-engine lever can recover without a fork.")
'''

MD_PLOT = """\
## 6 · Figure

Regenerated from `results/measured_t4.csv` alone, so it can be reproduced
without rerunning the GPU work.
"""

CODE_PLOT = '''\
import csv, statistics as st
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results/measured_t4.csv")))
modes = ["concurrent", "priority", "gated"]
labels = {"concurrent": "Concurrent\\n(default)",
          "priority": "vLLM priority",
          "gated": "Gated\\n(agency-preserving)"}
colors = {"concurrent": "#2563EB", "priority": "#D97706", "gated": "#BE185D"}

fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=160)
for ax in (a, b):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    ax.set_axisbelow(True)

def med(mode, key):
    v = [float(r[key]) for r in rows if r["mode"] == mode]
    return st.median(v) if v else 0.0

xs = range(len(modes))
a.bar(xs, [med(m, "order_tau") for m in modes],
      color=[colors[m] for m in modes], width=0.55, edgecolor="white", linewidth=2)
a.set_xticks(list(xs)); a.set_xticklabels([labels[m] for m in modes], fontsize=8.5)
a.set_ylabel("Order fidelity τ (measured)"); a.set_ylim(-0.2, 1.1)
a.axhline(0, color="#6b7280", linewidth=1, linestyle="--")
a.set_title("Designed turn order, measured on a T4", fontsize=10, loc="left")

b.bar(xs, [med(m, "gate_gap_s") for m in modes],
      color=[colors[m] for m in modes], width=0.55, edgecolor="white", linewidth=2)
b.axhline(1.2, color="#6b7280", linewidth=1.2, linestyle="--")
b.annotate("human reaction threshold", (len(modes) - 0.5, 1.24), fontsize=8,
           color="#6b7280", ha="right")
b.set_xticks(list(xs)); b.set_xticklabels([labels[m] for m in modes], fontsize=8.5)
b.set_ylabel("Human floor time at gates (s)")
b.set_title("Intervention window", fontsize=10, loc="left")

fig.tight_layout()
fig.savefig("results/figures/fig3_measured_t4.png", bbox_inches="tight",
            facecolor="white")
print("wrote results/figures/fig3_measured_t4.png")
'''

MD_END = """\
## 7 · Keep the outputs

Commit `results/measured_t4.csv`, `results/calibration_t4.json` and
`results/figures/fig3_measured_t4.png` back to the repository, and record in
the README **which GPU, which model, which vLLM version** produced them. A
measured result without its hardware provenance is not reproducible.
"""

CODE_END = '''\
import subprocess, json, datetime
prov = {
    "date_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
    "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                           "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip(),
    "model": MODEL,
    "vllm": subprocess.run(["python", "-c", "import vllm; print(vllm.__version__)"],
                           capture_output=True, text=True).stdout.strip(),
    "torch": subprocess.run(["python", "-c", "import torch; print(torch.__version__)"],
                            capture_output=True, text=True).stdout.strip(),
}
open("results/provenance_t4.json", "w").write(json.dumps(prov, indent=2))
print(json.dumps(prov, indent=2))

from google.colab import files       # noqa: E402
for f in ["results/measured_t4.csv", "results/calibration_t4.json",
          "results/provenance_t4.json",
          "results/figures/fig3_measured_t4.png"]:
    files.download(f)
'''


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.rstrip("\n").splitlines(keepends=True)}


def main() -> None:
    nb = {
        "cells": [
            md(MD_INTRO), md(MD_SETUP), code(CODE_GPU), code(CODE_INSTALL),
            md(MD_LAUNCH), code(CODE_LAUNCH),
            md(MD_CAL), code(CODE_CAL),
            md(MD_MEASURE), code(CODE_MEASURE),
            md(MD_COMPARE), code(CODE_COMPARE),
            md(MD_PLOT), code(CODE_PLOT),
            md(MD_END), code(CODE_END),
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {OUT} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
