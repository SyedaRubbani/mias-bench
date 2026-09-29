"""Mixed-initiative multi-agent workload model.

A *session* is one human analysis episode in a mixed-initiative visual
analytics system (the MIVAIS setting). The human takes an action; a group of
agents is expected to respond in a **designed turn order** that encodes the
interaction design (e.g. the Critic must speak after the Analyst so the human
can weigh the analysis before hearing the critique).

The point of the whole study is that this designed order is *advisory*: the
serving engine decides the realised order, and it does so on the basis of KV
block availability and queue mechanics, not interaction design.

Prompt structure is what makes prefix sharing non-trivial:

    [ deployment system prompt ][ agent persona ][ session context ][ turn tail ]
      shared by every agent       per agent role   shared in session  unique

So agents in one session share a long prefix, agents across sessions share a
shorter one, and an agent's *second* turn in a session shares almost
everything with its first. Which of those shares is resident when a turn is
admitted is what creates the timing lottery.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class AgentSpec:
    name: str
    persona_tokens: int
    output_tokens_mean: int
    # Position in the interaction design's intended turn order (0 = speaks first).
    designed_rank: int
    # True if the human is expected to be able to interrupt before this agent
    # speaks; these are the turns where displaced initiative actually costs
    # the human something.
    human_gate_before: bool = False


DEFAULT_AGENTS: List[AgentSpec] = [
    # Persona lengths are deliberately close to one another. If designed_rank
    # were correlated with prompt length, inversions would be a trivial
    # artefact of "short prompts prefill first". Keeping lengths comparable
    # forces any inversion to come from cache state and queue mechanics, which
    # is the effect under study.
    AgentSpec("analyst", persona_tokens=290, output_tokens_mean=160, designed_rank=0),
    AgentSpec("critic", persona_tokens=305, output_tokens_mean=150, designed_rank=1,
              human_gate_before=True),
    AgentSpec("retriever", persona_tokens=275, output_tokens_mean=145, designed_rank=2),
    AgentSpec("summariser", persona_tokens=298, output_tokens_mean=155, designed_rank=3,
              human_gate_before=True),
]


@dataclass
class Turn:
    """One LLM call: an agent responding within a session."""
    turn_id: int
    session_id: int
    round_idx: int
    agent: str
    designed_rank: int
    human_gate_before: bool
    token_ids: List[int]
    output_tokens: int
    arrival_time: float
    # Filled in by the engine.
    admit_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    preemptions: int = 0
    cached_blocks: int = 0
    computed_blocks: int = 0
    # Engine runtime state.
    prefilled_tokens: int = 0
    decoded_tokens: int = 0
    blocks: List[int] = field(default_factory=list)

    @property
    def ttft(self) -> Optional[float]:
        if self.first_token_time is None or self.arrival_time is None:
            return None
        return self.first_token_time - self.arrival_time


@dataclass
class Session:
    session_id: int
    turns: List[Turn] = field(default_factory=list)


class WorkloadGenerator:
    """Generates sessions with realistic prefix-sharing structure."""

    def __init__(
        self,
        seed: int,
        agents: Optional[List[AgentSpec]] = None,
        system_tokens: int = 512,
        context_tokens_mean: int = 900,
        tail_tokens_mean: int = 140,
        rounds_per_session: int = 2,
        dispatch_jitter: float = 0.005,
    ):
        self.rng = random.Random(seed)
        self.agents = agents or DEFAULT_AGENTS
        self.system_tokens = system_tokens
        self.context_tokens_mean = context_tokens_mean
        self.tail_tokens_mean = tail_tokens_mean
        self.rounds_per_session = rounds_per_session
        # Seconds of spread across a round's fan-out. Real multi-agent
        # frameworks dispatch a round asynchronously, so the server sees the
        # turns in an order set by network and event-loop timing, not by the
        # interaction design. Set to 0.0 for the ordered-dispatch control.
        self.dispatch_jitter = dispatch_jitter
        self._next_turn_id = 0
        # A fixed deployment-wide system prompt: identical token ids for all.
        self._system = [1_000 + i for i in range(system_tokens)]
        self._personas = {
            a.name: [2_000_000 + hash(a.name) % 1000 * 10_000 + i
                     for i in range(a.persona_tokens)]
            for a in self.agents
        }

    def _jitter(self, mean: int, spread: float = 0.25) -> int:
        return max(16, int(self.rng.gauss(mean, mean * spread)))

    def session(self, session_id: int, start_time: float) -> Session:
        ctx_len = self._jitter(self.context_tokens_mean)
        context = [5_000_000 + session_id * 100_000 + i for i in range(ctx_len)]
        sess = Session(session_id=session_id)
        t = start_time
        for _round in range(self.rounds_per_session):
            # All agents in a round are dispatched together: this is the
            # mixed-initiative "everyone reacts to the human's action" burst.
            # They arrive within a few ms of each other, so arrival order
            # carries no design intent -- only designed_rank does.
            for agent in sorted(self.agents, key=lambda a: a.designed_rank):
                tail_len = self._jitter(self.tail_tokens_mean)
                tail = [7_000_000 + self._next_turn_id * 1_000 + i
                        for i in range(tail_len)]
                tokens = self._system + self._personas[agent.name] + context + tail
                turn = Turn(
                    turn_id=self._next_turn_id,
                    session_id=session_id,
                    round_idx=_round,
                    agent=agent.name,
                    designed_rank=agent.designed_rank,
                    human_gate_before=agent.human_gate_before,
                    token_ids=tokens,
                    output_tokens=self._jitter(agent.output_tokens_mean),
                    arrival_time=t + self.rng.uniform(0.0, self.dispatch_jitter),
                )
                self._next_turn_id += 1
                sess.turns.append(turn)
            # The human reads and acts before the next round.
            t += self.rng.uniform(4.0, 9.0)
            # Subsequent rounds extend the shared session context.
            context = context + [5_000_000 + session_id * 100_000 + 50_000 + i
                                 for i in range(self._jitter(300))]
        return sess

    def generate(self, n_sessions: int, arrival_rate: float) -> List[Session]:
        """Poisson session arrivals at `arrival_rate` sessions/second."""
        sessions, t = [], 0.0
        for sid in range(n_sessions):
            t += self.rng.expovariate(arrival_rate)
            sessions.append(self.session(sid, t))
        return sessions
