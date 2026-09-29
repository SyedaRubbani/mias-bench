"""Live measurement against a real OpenAI-compatible LLM server.

This is the counterpart to the simulator: identical workload structure,
identical metrics, but the timings come from a real engine on real hardware
instead of a cost model. Anything in `mias/metrics.py` can score the output of
either, so simulated and measured results are directly comparable.

Three dispatch modes, matching the three policies in `mias/policies.py`:

  concurrent  all turns in a round are fired at once. This is what a real
              multi-agent framework does, and what the server's own scheduler
              then reorders. The H1 baseline.

  priority    fired at once, but each request carries vLLM's `priority` field
              set from the designed turn rank. Requires the server to run with
              --scheduling-policy priority. An in-engine lever that needs no
              fork, and a partial H3 arm.

  gated       turn k is not dispatched until turn k-1 has produced its first
              token plus `gate_seconds`. A client-side approximation of the
              agency-preserving policy: it guarantees order and reopens the
              human's window, and it pays for that in lost overlap. The full
              in-engine version would recover some of that cost; measuring the
              gap between the two is exactly the point.

Everything here is I/O, so it is tested against a mock server in
`tests/test_measure.py` rather than against a GPU.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .workload import DEFAULT_AGENTS, AgentSpec, Turn

# A small closed vocabulary keeps prompts realistic to tokenize while staying
# deterministic across runs.
_VOCAB = (
    "analysis cohort variance cluster baseline gradient residual segment "
    "threshold anomaly corpus feature pipeline inference estimate sample "
    "boundary weighting spectrum interval"
).split()


def _filler(n_words: int, rng: random.Random) -> str:
    return " ".join(rng.choice(_VOCAB) for _ in range(n_words))


@dataclass
class RoundSpec:
    """One round of agent turns, with the prompt text each will send."""
    session_id: int
    round_idx: int
    turns: List[Turn]
    prompts: Dict[int, str]


def build_rounds(
    n_sessions: int,
    rounds_per_session: int = 2,
    agents: Optional[Sequence[AgentSpec]] = None,
    system_words: int = 400,
    context_words: int = 700,
    tail_words: int = 120,
    max_tokens: int = 128,
    seed: int = 0,
) -> List[RoundSpec]:
    """Build rounds whose prompts share prefixes the way real agents do.

    Layout, longest shared segment first, so the server's prefix cache has
    something real to hit:

        [deployment system prompt][agent persona][session context][turn tail]
          same for every request    per agent      per session      unique
    """
    agents = list(agents or DEFAULT_AGENTS)
    rng = random.Random(seed)
    system = "SYSTEM. " + _filler(system_words, rng)
    personas = {
        a.name: f"ROLE {a.name}. " + _filler(140, rng) for a in agents
    }

    rounds: List[RoundSpec] = []
    turn_id = 0
    for sid in range(n_sessions):
        context = f"SESSION {sid}. " + _filler(context_words, rng)
        for r in range(rounds_per_session):
            turns: List[Turn] = []
            prompts: Dict[int, str] = {}
            for agent in sorted(agents, key=lambda a: a.designed_rank):
                tail = f"TURN {turn_id}. " + _filler(tail_words, rng)
                text = f"{system}\n{personas[agent.name]}\n{context}\n{tail}\nReply:"
                turns.append(Turn(
                    turn_id=turn_id,
                    session_id=sid,
                    round_idx=r,
                    agent=agent.name,
                    designed_rank=agent.designed_rank,
                    human_gate_before=agent.human_gate_before,
                    token_ids=[],           # unused on the live path
                    output_tokens=max_tokens,
                    arrival_time=0.0,
                ))
                prompts[turn_id] = text
                turn_id += 1
            rounds.append(RoundSpec(sid, r, turns, prompts))
            # Each round extends the shared session context.
            context = context + " " + _filler(200, rng)
    return rounds


# --------------------------------------------------------------------------
# Server interaction
# --------------------------------------------------------------------------
def parse_prometheus(text: str) -> Dict[str, float]:
    """Sum Prometheus samples by metric name, ignoring labels."""
    out: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition(" ")
        base = name.split("{", 1)[0]
        try:
            out[base] = out.get(base, 0.0) + float(rest.strip())
        except ValueError:
            continue
    return out


async def scrape(session, base_url: str) -> Dict[str, float]:
    try:
        async with session.get(f"{base_url}/metrics", timeout=10) as resp:
            return parse_prometheus(await resp.text())
    except Exception:
        return {}


async def _stream_one(
    session,
    base_url: str,
    model: str,
    turn: Turn,
    prompt: str,
    t_origin: float,
    priority: Optional[int] = None,
) -> None:
    """Send one completion request and record TTFT and completion time."""
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": turn.output_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    if priority is not None:
        payload["priority"] = priority
    turn.arrival_time = time.perf_counter() - t_origin
    first = True
    async with session.post(f"{base_url}/v1/completions", json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode("utf-8", "ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            if line == "data: [DONE]":
                break
            if first:
                turn.first_token_time = time.perf_counter() - t_origin
                first = False
    turn.finish_time = time.perf_counter() - t_origin


async def measure_round(
    session,
    base_url: str,
    model: str,
    spec: RoundSpec,
    mode: str = "concurrent",
    gate_seconds: float = 1.2,
    t_origin: Optional[float] = None,
) -> None:
    """Run one round in the given dispatch mode, filling in turn timings."""
    if mode not in ("concurrent", "priority", "gated"):
        raise ValueError(f"unknown mode {mode!r}")
    t_origin = t_origin if t_origin is not None else time.perf_counter()
    ordered = sorted(spec.turns, key=lambda t: t.designed_rank)

    if mode == "gated":
        # Serialise on the designed order, holding the floor open at gates.
        for turn in ordered:
            if turn.human_gate_before:
                await asyncio.sleep(gate_seconds)
            await _stream_one(session, base_url, model, turn,
                              spec.prompts[turn.turn_id], t_origin)
        return

    tasks = []
    for turn in ordered:
        prio = turn.designed_rank if mode == "priority" else None
        tasks.append(_stream_one(session, base_url, model, turn,
                                 spec.prompts[turn.turn_id], t_origin, prio))
    await asyncio.gather(*tasks)


async def run_measurement(
    base_url: str,
    model: str,
    rounds: Sequence[RoundSpec],
    mode: str = "concurrent",
    gate_seconds: float = 1.2,
    warmup_rounds: int = 1,
) -> Tuple[List[Turn], Dict[str, float]]:
    """Run every round sequentially in one mode; return turns and metric deltas.

    Rounds run one after another so that each round's ordering reflects its own
    contention rather than spillover from the previous one. Warmup rounds are
    executed and discarded: the first request to a cold server pays for CUDA
    graph capture and cache warm-up and is not representative.
    """
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for spec in rounds[:warmup_rounds]:
            await measure_round(session, base_url, model, spec, mode, gate_seconds)
        body = list(rounds[warmup_rounds:])
        before = await scrape(session, base_url)
        t_origin = time.perf_counter()
        for spec in body:
            await measure_round(session, base_url, model, spec, mode,
                                gate_seconds, t_origin)
        after = await scrape(session, base_url)

    keys = (
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:num_preemptions_total",
        "vllm:gpu_prefix_cache_queries_total",
        "vllm:gpu_prefix_cache_hits_total",
    )
    delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in keys if k in after}
    turns = [t for spec in body for t in spec.turns]
    return turns, delta


# --------------------------------------------------------------------------
# Calibration: fit the simulator's step-cost constants to this deployment
# --------------------------------------------------------------------------
async def calibrate(
    base_url: str,
    model: str,
    prompt_words: Sequence[int] = (200, 600, 1200, 2400),
    repeats: int = 6,
    seed: int = 0,
) -> Dict[str, float]:
    """Regress isolated, cache-cold TTFT on prompt length.

    Returns alpha (s per prompt word), beta (s of fixed overhead) and R^2. An
    R^2 below about 0.9 means the simulator's linear step model does not
    describe this deployment; report that rather than using the constants.
    """
    import aiohttp

    rng = random.Random(seed)
    xs: List[float] = []
    ys: List[float] = []
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for n_words in prompt_words:
            for rep in range(repeats):
                # A unique nonce defeats the prefix cache for this request.
                prompt = f"NONCE {time.time_ns()}-{rep} " + _filler(n_words, rng)
                turn = Turn(turn_id=-1, session_id=-1, round_idx=0, agent="cal",
                            designed_rank=0, human_gate_before=False,
                            token_ids=[], output_tokens=1, arrival_time=0.0)
                t0 = time.perf_counter()
                await _stream_one(session, base_url, model, turn, prompt, t0)
                if turn.first_token_time is None:
                    continue
                xs.append(float(n_words))
                ys.append(turn.first_token_time)

    n = len(xs)
    if n < 3:
        return {"alpha_s_per_word": 0.0, "beta_s": 0.0, "r2": 0.0, "n": float(n)}
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    alpha = sxy / sxx if sxx else 0.0
    beta = my - alpha * mx
    ss_res = sum((y - (alpha * x + beta)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return {
        "alpha_s_per_word": alpha,
        "beta_s": beta,
        "r2": 1 - ss_res / ss_tot if ss_tot else 0.0,
        "n": float(n),
    }
