"""Central writer for runs and steps: numbering, redaction and blobs.

The daemon owns step numbering. One lock serializes all writers, so step
indices and occurrences are assigned in arrival order.
"""

from __future__ import annotations

import secrets
import sqlite3
import threading
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from checkpoint.matching.normalize import MaskRule, match_key
from checkpoint.matching.redact import RedactionRule, redact_text, redact_value
from checkpoint.storage.cas import BlobStore

DEFAULT_LLM_STEP_NAME = "llm_call"

_INSERT_STEP_SQL = (
    "INSERT INTO steps (step_id, run_id, step_index, step_type, name, occurrence, match_key, "
    "input_hash, output_hash, post_state_hash, frozen_timestamp, rng_seed, tool_mode, source, "
    "error, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def boundary_seed(run_seed: int, step_index: int) -> int:
    """Deterministic per-boundary RNG seed, stored now and reused by replay."""
    return (run_seed * 6364136223846793005 + step_index) & 0x7FFFFFFFFFFFFFFF


def _total_tokens(output_obj: Any) -> int:
    if not isinstance(output_obj, Mapping):
        return 0
    usage = output_obj.get("usage")
    if not isinstance(usage, Mapping):
        return 0
    total = usage.get("total_tokens")
    return int(total) if isinstance(total, int) else 0


class Recorder:
    """Creates runs and persists steps with redaction and content-addressed blobs."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        blob_store: BlobStore,
        masks: tuple[MaskRule, ...] = (),
        redaction: tuple[RedactionRule, ...] = (),
    ) -> None:
        self._conn = conn
        self._blobs = blob_store
        self._masks = masks
        self._redaction = redaction
        self._lock = threading.Lock()
        self._next_index: dict[str, int] = {}

    def create_run(
        self,
        name: str,
        mode: str = "RECORD",
        parent_run_id: str | None = None,
        forked_at_step: int | None = None,
        variant_name: str | None = None,
    ) -> dict[str, Any]:
        run_id = uuid.uuid4().hex
        run_seed = secrets.randbits(63)
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (run_id, name, parent_run_id, variant_name, forked_at_step, "
                "mode, run_seed, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'RUNNING', ?)",
                (
                    run_id,
                    name,
                    parent_run_id,
                    variant_name,
                    forked_at_step,
                    mode,
                    run_seed,
                    time.time(),
                ),
            )
            self._conn.commit()
        return {"run_id": run_id, "run_seed": run_seed, "mode": mode}

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

    def finish_run(self, run_id: str, status: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))
            self._conn.commit()

    def record_llm_step(
        self,
        run_id: str,
        name: str,
        request_body: Any,
        response_body: Any,
        latency_ms: float,
        error: str | None = None,
    ) -> int:
        mk = match_key(request_body, self._masks)
        return self.record_step(
            run_id=run_id,
            step_type="LLM_CALL",
            name=name,
            match_key_value=mk,
            input_obj=request_body,
            output_obj=response_body,
            latency_ms=latency_ms,
            error=error,
        )

    def record_tool_step(
        self,
        run_id: str,
        name: str,
        input_obj: Any,
        output_obj: Any,
        tool_mode: str | None = None,
        error: str | None = None,
        latency_ms: float = 0.0,
    ) -> int:
        mk = match_key({"tool": name, "input": input_obj}, self._masks)
        return self.record_step(
            run_id=run_id,
            step_type="TOOL_CALL",
            name=name,
            match_key_value=mk,
            input_obj=input_obj,
            output_obj=output_obj,
            tool_mode=tool_mode,
            error=error,
            latency_ms=latency_ms,
        )

    def record_checkpoint(self, run_id: str, name: str, state: Any) -> int:
        mk = match_key({"checkpoint": name}, self._masks)
        return self.record_step(
            run_id=run_id,
            step_type="CHECKPOINT",
            name=name,
            match_key_value=mk,
            input_obj={},
            output_obj=state,
            post_state_obj=state,
        )

    def record_step(
        self,
        *,
        run_id: str,
        step_type: str,
        name: str,
        match_key_value: str,
        input_obj: Any,
        output_obj: Any = None,
        post_state_obj: Any = None,
        tool_mode: str | None = None,
        error: str | None = None,
        latency_ms: float = 0.0,
    ) -> int:
        with self._lock:
            run = self.get_run(run_id)
            if run is None:
                raise KeyError(f"unknown run: {run_id}")
            step_index = self._next_index.get(run_id)
            if step_index is None:
                row = self._conn.execute(
                    "SELECT COALESCE(MAX(step_index) + 1, 0) AS next FROM steps WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                step_index = int(row["next"])
            occurrence = int(
                self._conn.execute(
                    "SELECT COUNT(*) AS n FROM steps WHERE run_id = ? AND name = ?",
                    (run_id, name),
                ).fetchone()["n"]
            )
            input_hash = self._blobs.put_json(redact_value(input_obj, self._redaction))
            output_hash = (
                self._blobs.put_json(redact_value(output_obj, self._redaction))
                if output_obj is not None
                else None
            )
            post_state_hash = (
                self._blobs.put_json(redact_value(post_state_obj, self._redaction))
                if post_state_obj is not None
                else None
            )
            self._conn.execute(
                _INSERT_STEP_SQL,
                (
                    uuid.uuid4().hex,
                    run_id,
                    step_index,
                    step_type,
                    name,
                    occurrence,
                    match_key_value,
                    input_hash,
                    output_hash,
                    post_state_hash,
                    _now_iso(),
                    boundary_seed(int(run["run_seed"]), step_index),
                    tool_mode,
                    "RECORDED",
                    redact_text(error, self._redaction) if error else None,
                    latency_ms,
                ),
            )
            self._conn.execute(
                "UPDATE runs SET total_tokens = total_tokens + ?, "
                "total_latency_ms = total_latency_ms + ? WHERE run_id = ?",
                (_total_tokens(output_obj), latency_ms, run_id),
            )
            self._conn.commit()
            self._next_index[run_id] = step_index + 1
            return step_index
