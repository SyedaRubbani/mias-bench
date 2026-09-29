"""Discrete-event model of a continuous-batching LLM serving engine.

Each engine step:
  1. the policy orders the waiting queue and gates individual turns;
  2. turns are admitted while KV blocks and the token budget allow;
  3. admitted turns are prefilled in chunks (chunked prefill), interleaved
     with one decode token for every already-running turn;
  4. if a decoding turn needs a new block and the pool is exhausted, the
     policy nominates a victim, which is preempted and requeued.

Step duration follows a two-parameter linear model
    t_step = alpha * batched_tokens + beta
whose constants live in EngineConfig and are documented in README.md as
placeholders to be replaced by measurements from a real engine through
`mias.adapters.vllm_adapter`. Nothing in the findings depends on their exact
values; the experiments report sensitivity to them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .kv import BlockAllocator
from .policies import Policy
from .provenance import ProvenanceLog
from .workload import Session, Turn


@dataclass
class EngineConfig:
    total_blocks: int = 2048           # KV pool size, in blocks
    block_size: int = 16
    max_batched_tokens: int = 2048     # per-step token budget
    max_running: int = 24              # max concurrent sequences
    chunk_size: int = 512              # max prefill tokens per request per step
    alpha_s_per_token: float = 1.2e-4  # marginal step cost per batched token
    beta_s: float = 8.0e-3             # fixed per-step overhead
    decode_blocks_reserved: int = 8    # blocks reserved up front for decoding


class Engine:
    def __init__(
        self,
        config: EngineConfig,
        policy: Policy,
        log: Optional[ProvenanceLog] = None,
    ):
        self.cfg = config
        self.policy = policy
        self.allocator = BlockAllocator(config.total_blocks, config.block_size)
        self.log = log or ProvenanceLog()
        self.now = 0.0
        self.steps = 0
        self.total_batched_tokens = 0

    def run(self, sessions: Sequence[Session]) -> List[Turn]:
        pending: List[Turn] = sorted(
            (t for s in sessions for t in s.turns),
            key=lambda t: (t.arrival_time, t.turn_id),
        )
        turns_by_key = {
            (t.session_id, t.round_idx, t.designed_rank): t for t in pending
        }
        all_turns = list(pending)
        waiting: List[Turn] = []
        running: List[Turn] = []
        first_token: Dict[int, float] = {}
        done = 0
        total = len(pending)
        guard = 0

        while done < total:
            guard += 1
            if guard > 2_000_000:
                raise RuntimeError("engine did not converge; check policy gating")

            # Release newly arrived turns into the waiting queue.
            while pending and pending[0].arrival_time <= self.now:
                t = pending.pop(0)
                waiting.append(t)
                self.log.emit("turn_arrived", self.now, t)

            if not waiting and not running:
                if pending:
                    self.now = pending[0].arrival_time
                    continue
                break

            # --- admission -------------------------------------------------
            for turn in self.policy.order_waiting(
                waiting, self.allocator, self.now, first_token
            ):
                if len(running) >= self.cfg.max_running:
                    break
                if not self.policy.admissible(
                    turn, self.now, first_token, turns_by_key
                ):
                    continue
                res, blocks = self.allocator.allocate(
                    turn.token_ids, self.cfg.decode_blocks_reserved, self.now
                )
                if not res.ok:
                    victim = self.policy.preemption_victim(running)
                    if victim is not None and victim.turn_id != turn.turn_id:
                        self._preempt(victim, running, waiting)
                    break
                turn.blocks = blocks
                turn.cached_blocks = res.cached_blocks
                turn.computed_blocks = res.computed_blocks
                turn.admit_time = self.now
                # Cached prefix blocks need no recompute: this is the whole
                # source of the timing lottery we are measuring.
                turn.prefilled_tokens = res.cached_blocks * self.cfg.block_size
                waiting.remove(turn)
                running.append(turn)
                self.log.emit(
                    "turn_admitted", self.now, turn,
                    cached_blocks=res.cached_blocks,
                    computed_blocks=res.computed_blocks,
                    kv_utilisation=round(self.allocator.utilisation, 4),
                )

            if not running:
                # Everything is gated or starved; advance to the next event.
                nxt = [t.arrival_time for t in pending[:1]]
                gate_times = [
                    ft + getattr(self.policy, "gate_seconds", 0.0)
                    for ft in first_token.values()
                ]
                future = [x for x in nxt + gate_times if x > self.now]
                self.now = min(future) if future else self.now + 0.01
                continue

            # --- one engine step -------------------------------------------
            budget = self.cfg.max_batched_tokens
            batched = 0

            # Decode first: running sequences that have finished prefill.
            decoders = [t for t in running if t.prefilled_tokens >= len(t.token_ids)]
            for t in decoders:
                if budget <= 0:
                    break
                budget -= 1
                batched += 1

            # Then chunked prefill for the rest.
            prefillers = [t for t in running if t.prefilled_tokens < len(t.token_ids)]
            for t in prefillers:
                if budget <= 0:
                    break
                take = min(
                    self.cfg.chunk_size,
                    budget,
                    len(t.token_ids) - t.prefilled_tokens,
                )
                t.prefilled_tokens += take
                budget -= take
                batched += take

            dt = self.cfg.alpha_s_per_token * batched + self.cfg.beta_s
            self.now += dt
            self.steps += 1
            self.total_batched_tokens += batched

            # Emit first tokens for anything that just finished prefill.
            for t in prefillers:
                if t.prefilled_tokens >= len(t.token_ids) and t.turn_id not in first_token:
                    t.first_token_time = self.now
                    first_token[t.turn_id] = self.now
                    self.log.emit("first_token", self.now, t, ttft=round(t.ttft, 4))

            # Advance decoders; grow KV as they go.
            for t in list(decoders):
                t.decoded_tokens += 1
                if t.decoded_tokens % self.cfg.block_size == 0:
                    res, blocks = self.allocator.allocate([], 1, self.now)
                    if not res.ok:
                        victim = self.policy.preemption_victim(running) or t
                        self._preempt(victim, running, waiting)
                        continue
                    t.blocks.extend(blocks)
                if t.decoded_tokens >= t.output_tokens:
                    t.finish_time = self.now
                    self.allocator.free(t.blocks)
                    t.blocks = []
                    running.remove(t)
                    done += 1
                    self.log.emit("turn_finished", self.now, t,
                                  latency=round(t.finish_time - t.arrival_time, 4))

        return all_turns

    def _preempt(self, victim: Turn, running: List[Turn], waiting: List[Turn]) -> None:
        self.allocator.free(victim.blocks)
        victim.blocks = []
        victim.prefilled_tokens = 0
        victim.decoded_tokens = 0
        victim.admit_time = None
        victim.preemptions += 1
        if victim in running:
            running.remove(victim)
        waiting.append(victim)
        self.log.emit("turn_preempted", self.now, victim,
                      preemptions=victim.preemptions,
                      kv_utilisation=round(self.allocator.utilisation, 4))
