"""Serving provenance: the contribution that makes the rest analysable.

Mixed-initiative systems already log *interaction* provenance - what the human
clicked, when, and what the agents said. That log cannot explain why agent B
answered before agent A, because the cause sits one layer down.

This module emits a JSONL stream in the same shape as an interaction
provenance log, so the two can be joined on (session_id, round_idx, timestamp)
and analysed together. Every record answers: which turn, at what wall-clock
offset, in what engine state.

Schema (one JSON object per line):

    event            str   one of turn_arrived | turn_admitted | first_token
                           | turn_preempted | turn_finished
    t                float seconds since session start
    turn_id          int
    session_id       int
    round_idx        int
    agent            str
    designed_rank    int
    human_gate_before bool
    prompt_tokens    int
    output_tokens    int
    <extras>               event-specific: ttft, cached_blocks,
                           computed_blocks, kv_utilisation, preemptions,
                           latency

Deliberately flat and boring: it should be loadable with
`pandas.read_json(path, lines=True)` by someone who has never seen this code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"

EVENTS = (
    "turn_arrived",
    "turn_admitted",
    "first_token",
    "turn_preempted",
    "turn_finished",
)


class ProvenanceLog:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.records: List[Dict[str, Any]] = []

    def emit(self, event: str, t: float, turn, **extras: Any) -> None:
        if not self.enabled:
            return
        assert event in EVENTS, f"unknown event {event!r}"
        rec: Dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "event": event,
            "t": round(t, 6),
            "turn_id": turn.turn_id,
            "session_id": turn.session_id,
            "round_idx": turn.round_idx,
            "agent": turn.agent,
            "designed_rank": turn.designed_rank,
            "human_gate_before": turn.human_gate_before,
            "prompt_tokens": len(turn.token_ids),
            "output_tokens": turn.output_tokens,
        }
        rec.update(extras)
        self.records.append(rec)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for rec in self.records:
                fh.write(json.dumps(rec) + "\n")
        return path

    @staticmethod
    def read(path: str | Path) -> List[Dict[str, Any]]:
        with Path(path).open() as fh:
            return [json.loads(line) for line in fh if line.strip()]
