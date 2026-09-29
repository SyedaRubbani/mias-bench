"""Adapter for running the same experiment against a live vLLM server.

The simulator answers "does this mechanism exist and can it be controlled".
It cannot answer "what are the real constants". This adapter is the path from
one to the other, and it is the first thing to run on a machine with a GPU.

It is deliberately thin and deliberately untested in this repository: no GPU
was available when the results in results/ were produced, and claiming
otherwise would be dishonest. Treat every function below as a specification
with a reference implementation, not as validated code.

Two jobs:

1. `calibrate()` - measure the engine's real step-cost constants (alpha, beta)
   and hand back an EngineConfig, so the simulator stops using placeholders.

2. `run_round()` - dispatch one round of agent turns concurrently against a
   live server and record the same provenance events the simulator emits, so
   `mias.metrics` can score real runs and simulated runs with identical code.

Serving-side ground truth (cache hits, preemptions, queue depth) comes from
the engine's own Prometheus metrics rather than from client-side timing:

    vllm:prefix_cache_queries_total / vllm:prefix_cache_hits_total
    vllm:num_preemptions_total
    vllm:num_requests_waiting / vllm:num_requests_running
    vllm:gpu_cache_usage_perc

Start the server with prefix caching on and metrics exposed, e.g.

    vllm serve <model> --enable-prefix-caching --max-num-seqs 24 \
        --max-num-batched-tokens 2048 --enable-chunked-prefill

Then:

    python -m mias.adapters.vllm_adapter --base-url http://localhost:8000 \
        --model <model> --calibrate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

from ..engine import EngineConfig
from ..provenance import ProvenanceLog
from ..workload import Turn


# --------------------------------------------------------------------------
# Metrics scraping
# --------------------------------------------------------------------------
def parse_prometheus(text: str) -> Dict[str, float]:
    """Minimal Prometheus text-format parser; keeps counters and gauges."""
    out: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition(" ")
        base = name.split("{", 1)[0]
        try:
            value = float(rest.strip())
        except ValueError:
            continue
        out[base] = out.get(base, 0.0) + value
    return out


async def scrape(session, base_url: str) -> Dict[str, float]:
    async with session.get(f"{base_url}/metrics") as resp:
        return parse_prometheus(await resp.text())


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
async def calibrate(
    base_url: str,
    model: str,
    prompt_lengths: Sequence[int] = (256, 1024, 2048, 4096),
    repeats: int = 5,
) -> Dict[str, float]:
    """Fit t_step = alpha * batched_tokens + beta by ordinary least squares.

    Sends single, isolated, cache-cold requests of varying prompt length and
    regresses TTFT on prompt tokens. Cache-cold is enforced by prefixing a
    unique nonce, which defeats the prefix cache for that request.

    Returns alpha (s/token), beta (s), and the R^2 of the fit. An R^2 below
    about 0.9 means the linear step model does not describe this deployment
    and the simulator's timing should not be trusted for it - report that
    rather than using the numbers anyway.
    """
    import aiohttp  # imported lazily; not a dependency of the simulator

    xs: List[float] = []
    ys: List[float] = []
    async with aiohttp.ClientSession() as session:
        for n_tokens in prompt_lengths:
            for rep in range(repeats):
                nonce = f"[{time.time_ns()}-{rep}] "
                prompt = nonce + " ".join(["token"] * n_tokens)
                t0 = time.perf_counter()
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 1,
                    "stream": True,
                    "temperature": 0.0,
                }
                async with session.post(
                    f"{base_url}/v1/completions", json=payload
                ) as resp:
                    async for _chunk in resp.content:
                        break                       # first streamed chunk = TTFT
                xs.append(float(n_tokens))
                ys.append(time.perf_counter() - t0)

    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    alpha = sxy / sxx if sxx else 0.0
    beta = my - alpha * mx
    ss_res = sum((y - (alpha * x + beta)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return {"alpha_s_per_token": alpha, "beta_s": beta, "r2": r2, "n": float(n)}


def config_from_calibration(
    base: EngineConfig, fit: Dict[str, float]
) -> EngineConfig:
    return replace(
        base,
        alpha_s_per_token=fit["alpha_s_per_token"],
        beta_s=fit["beta_s"],
    )


# --------------------------------------------------------------------------
# Live round execution
# --------------------------------------------------------------------------
async def run_round(
    base_url: str,
    model: str,
    turns: Sequence[Turn],
    prompts: Dict[int, str],
    log: ProvenanceLog,
    t_origin: Optional[float] = None,
) -> None:
    """Dispatch one round concurrently and record first-token times.

    This is the measurement that matters: the turns are fired together, the
    way a real multi-agent framework fans out a round, and we record the order
    in which the server actually starts answering them.
    """
    import aiohttp

    t_origin = t_origin if t_origin is not None else time.perf_counter()

    async def one(session, turn: Turn) -> None:
        log.emit("turn_arrived", time.perf_counter() - t_origin, turn)
        payload = {
            "model": model,
            "prompt": prompts[turn.turn_id],
            "max_tokens": turn.output_tokens,
            "stream": True,
            "temperature": 0.0,
        }
        first = True
        async with session.post(f"{base_url}/v1/completions", json=payload) as resp:
            async for chunk in resp.content:
                if not chunk.strip():
                    continue
                if first:
                    now = time.perf_counter() - t_origin
                    turn.first_token_time = now
                    log.emit("first_token", now, turn, ttft=round(now, 4))
                    first = False
        now = time.perf_counter() - t_origin
        turn.finish_time = now
        log.emit("turn_finished", now, turn, latency=round(now, 4))

    async with aiohttp.ClientSession() as session:
        before = await scrape(session, base_url)
        await asyncio.gather(*(one(session, t) for t in turns))
        after = await scrape(session, base_url)

    delta = {
        k: after.get(k, 0.0) - before.get(k, 0.0)
        for k in (
            "vllm:prefix_cache_queries_total",
            "vllm:prefix_cache_hits_total",
            "vllm:num_preemptions_total",
        )
    }
    queries = delta.get("vllm:prefix_cache_queries_total", 0.0)
    for turn in turns:
        turn.preemptions = int(delta.get("vllm:num_preemptions_total", 0.0))
        if queries:
            turn.cached_blocks = int(delta.get("vllm:prefix_cache_hits_total", 0.0))
            turn.computed_blocks = int(queries - turn.cached_blocks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out", default="results/calibration.json")
    args = ap.parse_args()

    if args.calibrate:
        fit = asyncio.run(calibrate(args.base_url, args.model))
        print(json.dumps(fit, indent=2))
        if fit["r2"] < 0.9:
            print("\nWARNING: R^2 < 0.9 - the linear step model does not "
                  "describe this deployment. Report this rather than using "
                  "the fitted constants.")
        with open(args.out, "w") as fh:
            json.dump(fit, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
