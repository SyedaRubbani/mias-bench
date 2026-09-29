#!/usr/bin/env python3
"""One command that runs the whole GPU measurement. No notebook required.

    python scripts/run_t4.py                    # full run, ~30 min on a T4
    python scripts/run_t4.py --smoke            # ~3 min, proves the path works
    python scripts/run_t4.py --probe-only       # just print supported flags

Why a script rather than notebook cells: cell ordering, stale saved copies and
Colab's sys.path all conspire to break a multi-cell flow in ways that have
nothing to do with the research. This does the same work in one process, with
one exit code, and writes the same files.

Outputs (all under results/):
    measured_t4.csv          one row per condition x mode x repeat
    calibration_t4.json      fitted step-cost constants and their R^2
    provenance_t4.json       GPU, model, vLLM version, flags actually used
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import functools
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Make the repo importable however this script was invoked.
REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from mias.measure import build_rounds, run_measurement, calibrate  # noqa: E402
from mias.metrics import summarise  # noqa: E402

SERVER_MODULE = "vllm.entrypoints.openai.api_server"
BASE_URL = "http://127.0.0.1:8000"
RESULTS = REPO_DIR / "results"
LOG_PATH = Path("/tmp/vllm_mias.log")

_server: subprocess.Popen | None = None
DROPPED_FLAGS: list[str] = []


# --------------------------------------------------------------------------
# Server control
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def supported_flags() -> frozenset[str]:
    """Flags this vLLM build accepts, read from its own --help.

    vLLM's CLI changes between releases. Probing beats hard-coding: an
    unknown flag makes argparse exit before the GPU is ever touched.
    """
    out = subprocess.run([sys.executable, "-m", SERVER_MODULE, "--help"],
                         capture_output=True, text=True, timeout=600)
    return frozenset(re.findall(r"(--[a-z0-9][a-z0-9-]*)", out.stdout + out.stderr))


def filter_flags(cmd: list[str]) -> list[str]:
    """Drop unsupported flags along with their values."""
    ok = supported_flags()
    kept: list[str] = []
    i = 0
    while i < len(cmd):
        tok = cmd[i]
        if tok.startswith("--") and tok not in ok:
            DROPPED_FLAGS.append(tok)
            i += 1
            if i < len(cmd) and not cmd[i].startswith("--"):
                i += 1
            continue
        kept.append(tok)
        i += 1
    return kept


def shutdown() -> None:
    global _server
    if _server is not None and _server.poll() is None:
        os.killpg(os.getpgid(_server.pid), signal.SIGTERM)
        try:
            _server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(_server.pid), signal.SIGKILL)
    _server = None
    time.sleep(5)


def launch(model: str, blocks: int | None = None, priority: bool = False,
           max_num_seqs: int = 16, max_model_len: int = 4096,
           timeout: int = 1200) -> None:
    global _server
    shutdown()
    cmd = filter_flags([
        sys.executable, "-m", SERVER_MODULE,
        "--model", model,
        "--dtype", "half",                 # Turing (SM 7.5) has no bfloat16
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--gpu-memory-utilization", "0.85",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--port", "8000",
    ] + (["--num-gpu-blocks-override", str(blocks)] if blocks else [])
      + (["--scheduling-policy", "priority"] if priority else []))

    with LOG_PATH.open("w") as log:
        _server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                   preexec_fn=os.setsid)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _server.poll() is not None:
            sys.stderr.write(LOG_PATH.read_text()[-4000:] + "\n")
            raise RuntimeError(
                "vLLM exited during startup (log tail above). If it names an "
                "unsupported attention backend on Turing, retry with "
                "VLLM_ATTENTION_BACKEND=TRITON_ATTN in the environment."
            )
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2) as r:
                if r.status == 200:
                    print(f"  server up in {time.time() - t0:.0f}s "
                          f"(blocks={blocks}, priority={priority})", flush=True)
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(3)
    raise TimeoutError(f"vLLM never became healthy; see {LOG_PATH}")


def kv_blocks_reported() -> str:
    if not LOG_PATH.exists():
        return "unknown"
    m = re.findall(r"GPU (?:KV cache size|blocks)[:=] *([\d,]+)",
                   LOG_PATH.read_text())
    return m[-1] if m else "unknown"


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
async def measure_all(args) -> list[dict]:
    rows: list[dict] = []
    conditions = ([("kv_large", None)] if args.smoke
                  else [("kv_large", None), ("kv_tight", args.tight_blocks)])
    modes = ["concurrent"] if args.smoke else ["concurrent", "priority", "gated"]
    repeats = 1 if args.smoke else args.repeats
    sessions = 2 if args.smoke else args.sessions

    for cond, blocks in conditions:
        launch(args.model, blocks=blocks, priority=True)
        print(f"  {cond}: KV blocks reported = {kv_blocks_reported()}", flush=True)
        for mode in modes:
            for rep in range(repeats):
                rounds = build_rounds(
                    n_sessions=sessions, rounds_per_session=2,
                    max_tokens=args.max_tokens, seed=1000 * rep,
                )
                turns, delta = await run_measurement(
                    BASE_URL, args.model, rounds, mode=mode,
                    gate_seconds=args.gate, warmup_rounds=1,
                )
                row = {"source": "measured_t4", "config": cond,
                       "mode": mode, "repeat": rep}
                row.update(summarise(turns, args.gate))
                row.update({k.replace("vllm:", ""): v for k, v in delta.items()})
                rows.append(row)
                print(f"    {cond:<9} {mode:<11} rep{rep}  "
                      f"tau={row['order_tau']:+.2f}  "
                      f"gap={row['gate_gap_s']:.2f}s  "
                      f"ttft={row['ttft_mean']:.2f}s  "
                      f"thru={row['throughput_tok_s']:.0f} tok/s", flush=True)
    shutdown()
    return rows


def write_outputs(rows: list[dict], fit: dict, args) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with (RESULTS / "measured_t4.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    (RESULTS / "calibration_t4.json").write_text(json.dumps(fit, indent=2))

    def _v(mod):
        return subprocess.run([sys.executable, "-c",
                               f"import {mod}; print({mod}.__version__)"],
                              capture_output=True, text=True).stdout.strip()

    prov = {
        "date_utc": datetime.datetime.now(datetime.timezone.utc)
                        .isoformat(timespec="seconds"),
        "gpu": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip(),
        "model": args.model,
        "vllm": _v("vllm"),
        "torch": _v("torch"),
        "server_flags_dropped": sorted(set(DROPPED_FLAGS)),
        "smoke_run": bool(args.smoke),
    }
    (RESULTS / "provenance_t4.json").write_text(json.dumps(prov, indent=2))
    print("\nwrote:")
    for name in ("measured_t4.csv", "calibration_t4.json", "provenance_t4.json"):
        print(f"  results/{name}")
    print("\nprovenance:", json.dumps(prov, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--gate", type=float, default=1.2,
                    help="human reaction threshold, seconds")
    ap.add_argument("--tight-blocks", type=int, default=512,
                    help="KV blocks for the kv_tight condition")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to prove the path works end to end")
    ap.add_argument("--probe-only", action="store_true",
                    help="print which server flags this vLLM build accepts")
    args = ap.parse_args()

    print("probing vLLM CLI ...", flush=True)
    flags = supported_flags()
    print(f"{len(flags)} flags supported by this build")
    critical = {"--num-gpu-blocks-override": "kv_tight condition",
                "--scheduling-policy": "priority arm"}
    missing = []
    for flag in ["--dtype", "--enable-prefix-caching", "--enable-chunked-prefill",
                 "--num-gpu-blocks-override", "--scheduling-policy"]:
        have = flag in flags
        note = "" if have else f"   <-- will drop ({critical.get(flag, 'cosmetic')})"
        print(f"  {flag:<32}{'yes' if have else 'NO'}{note}")
        if not have and flag in critical:
            missing.append(flag)
    if missing:
        print("\nNOTE: " + ", ".join(missing) + " unavailable in this build. "
              "The run continues, but the affected arm is not measured - say so "
              "in the write-up rather than quietly reporting fewer conditions.")
    if args.probe_only:
        return 0

    try:
        print("\ncalibrating ...", flush=True)
        launch(args.model)
        fit = asyncio.run(calibrate(
            BASE_URL, args.model,
            prompt_words=(200, 600) if args.smoke else (200, 600, 1200, 2400),
            repeats=2 if args.smoke else 6))
        print("  " + json.dumps(fit))
        if fit["r2"] < 0.9:
            print("  WARNING: R^2 < 0.9 - the simulator's linear step model does "
                  "not describe this deployment. Report that; do not quietly "
                  "use the fitted constants.")

        print("\nmeasuring ...", flush=True)
        rows = asyncio.run(measure_all(args))
        write_outputs(rows, fit, args)
    finally:
        shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
