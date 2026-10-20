"""Replay lookup by (name, occurrence) and divergence diffs.

Steps are matched by name and occurrence, not by global step index, so a
prompt edit early in the run does not invalidate everything after it.
The recorded match_key must still agree, otherwise the request itself changed
and the user needs to see a readable diff.
"""

from __future__ import annotations

import difflib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from checkpoint.matching.normalize import MaskRule, normalize_request


@dataclass(frozen=True)
class LookupResult:
    """Outcome of looking up a boundary call in the parent run."""

    found: bool
    matched: bool
    occurrence: int
    step: sqlite3.Row | None
    diff: str | None = None
    semantic: bool = False
    similarity: float | None = None
    recorded_input: Any = None

    @property
    def step_index(self) -> int | None:
        return int(self.step["step_index"]) if self.step is not None else None

    @property
    def frozen_timestamp(self) -> str | None:
        return str(self.step["frozen_timestamp"]) if self.step is not None else None

    @property
    def rng_seed(self) -> int | None:
        return int(self.step["rng_seed"]) if self.step is not None else None

    @property
    def error(self) -> str | None:
        return str(self.step["error"]) if self.step is not None and self.step["error"] else None


def find_recorded_step(
    conn: sqlite3.Connection, parent_run_id: str, name: str, occurrence: int
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM steps WHERE run_id = ? AND name = ? AND occurrence = ? "
        "ORDER BY step_index LIMIT 1",
        (parent_run_id, name, occurrence),
    ).fetchone()


def step_label(step_type: str, name: str, occurrence: int) -> str:
    kind = {"LLM_CALL": "LLM call", "TOOL_CALL": "tool call", "CHECKPOINT": "checkpoint"}.get(
        step_type, step_type
    )
    return f'{kind} "{name}" occurrence {occurrence}'


def normalized_diff(
    recorded_input: Any,
    incoming_input: Any,
    masks: tuple[MaskRule, ...],
    label: str,
) -> str:
    """Readable unified diff of two normalized boundary requests."""
    old = json.dumps(
        normalize_request(_as_mapping(recorded_input), masks),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    new = json.dumps(
        normalize_request(_as_mapping(incoming_input), masks),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    lines = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        fromfile=f"recorded {label}",
        tofile=f"replayed {label}",
        lineterm="",
        n=3,
    )
    return "\n".join(lines)


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {"value": value}
