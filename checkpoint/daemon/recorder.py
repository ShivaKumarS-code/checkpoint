"""Central writer for runs and steps: numbering, redaction and blobs.

The daemon owns step numbering. One lock serializes all writers, so step
indices and occurrences are assigned in arrival order.

In REPLAY mode steps are served from the parent run by ``(name, occurrence)``
and recorded with the parent's frozen timestamp and RNG seed, so the clock and
entropy stay on the recorded path.
"""

from __future__ import annotations

import os
import secrets
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from checkpoint.daemon.payloads import experiment_payload, run_payload
from checkpoint.matching.normalize import MaskRule, canonical_json, match_key
from checkpoint.matching.redact import RedactionRule, redact_text, redact_value
from checkpoint.matching.replay import (
    LookupResult,
    find_recorded_step,
    normalized_diff,
    step_label,
)
from checkpoint.sdk.determinism import boundary_seed, real_now_iso
from checkpoint.storage.cas import BlobStore

DEFAULT_LLM_STEP_NAME = "llm_call"

_REAL_URANDOM = os.urandom


def _new_id() -> str:
    """Opaque id from the real urandom.

    ``uuid.uuid4`` reads ``os.urandom`` at call time, so the SDK determinism
    patch would otherwise make run and step ids deterministic (and collide).
    """
    return uuid.UUID(bytes=_REAL_URANDOM(16), version=4).hex

_INSERT_STEP_SQL = (
    "INSERT INTO steps (step_id, run_id, step_index, step_type, name, occurrence, match_key, "
    "input_hash, output_hash, post_state_hash, frozen_timestamp, rng_seed, tool_mode, source, "
    "error, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


@dataclass(frozen=True)
class StepInfo:
    """Metadata of a recorded step that the SDK needs to settle its clock."""

    step_index: int
    occurrence: int
    frozen_timestamp: str
    rng_seed: int


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
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._conn = conn
        self._blobs = blob_store
        self._masks = masks
        self._redaction = redaction
        self._on_event = on_event
        self._lock = threading.Lock()
        self._next_index: dict[str, int] = {}

    def _emit(self, event: dict[str, Any]) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception:
            pass

    def create_run(
        self,
        name: str,
        mode: str = "RECORD",
        parent_run_id: str | None = None,
        forked_at_step: int | None = None,
        variant_name: str | None = None,
        experiment_id: str | None = None,
        run_seed: int | None = None,
    ) -> dict[str, Any]:
        run_id = _new_id()
        seed = secrets.randbits(63) if run_seed is None else run_seed
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (run_id, name, parent_run_id, experiment_id, variant_name, "
                "forked_at_step, mode, run_seed, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'RUNNING', ?)",
                (
                    run_id,
                    name,
                    parent_run_id,
                    experiment_id,
                    variant_name,
                    forked_at_step,
                    mode,
                    seed,
                    time.time(),
                ),
            )
            self._conn.commit()
        run = self.get_run(run_id)
        if run is not None:
            self._emit({"type": "run_started", "run": run_payload(run)})
        return {"run_id": run_id, "run_seed": seed, "mode": mode}

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

    def root_run(self, run_id: str) -> sqlite3.Row | None:
        """Walk parent links to the run whose start time seeds the frozen clock."""
        run = self.get_run(run_id)
        seen: set[str] = set()
        while run is not None and run["parent_run_id"] is not None:
            parent_id = str(run["parent_run_id"])
            if parent_id in seen:
                break
            seen.add(parent_id)
            parent = self.get_run(parent_id)
            if parent is None:
                break
            run = parent
        return run

    def finish_run(self, run_id: str, status: str) -> None:
        """Set a terminal status unless a divergence was already recorded."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE runs SET status = ? WHERE run_id = ? AND status = 'RUNNING'",
                (status, run_id),
            )
            self._conn.commit()
        if cursor.rowcount:
            run = self.get_run(run_id)
            if run is not None:
                self._emit({"type": "run_finished", "run": run_payload(run)})

    def mark_diverged(self, run_id: str, step_index: int | None, diff: str) -> None:
        """Record the first divergence and its readable diff."""
        with self._lock:
            run = self.get_run(run_id)
            if run is None:
                raise KeyError(f"unknown run: {run_id}")
            if run["status"] == "DIVERGED":
                return
            diff_hash = self._blobs.put_json({"diff": redact_text(diff, self._redaction)})
            self._conn.execute(
                "UPDATE runs SET status = 'DIVERGED', divergence_step = ?, "
                "divergence_diff_hash = ? WHERE run_id = ?",
                (step_index, diff_hash, run_id),
            )
            self._conn.commit()
        run = self.get_run(run_id)
        if run is not None:
            self._emit({"type": "run_finished", "run": run_payload(run)})

    def divergence_diff(self, run_id: str) -> str | None:
        run = self.get_run(run_id)
        if run is None or run["divergence_diff_hash"] is None:
            return None
        payload = self._blobs.get_json(str(run["divergence_diff_hash"]))
        if isinstance(payload, Mapping) and isinstance(payload.get("diff"), str):
            return str(payload["diff"])
        return None

    def create_experiment(
        self,
        name: str,
        parent_run_id: str,
        fork_step_index: int,
        variants: list[Any],
        checks: list[Any] | None = None,
    ) -> str:
        """Persist an experiment definition and return its id."""
        variants_hash = self._blobs.put_json({"variants": variants})
        checks_hash = self._blobs.put_json({"checks": checks}) if checks else None
        experiment_id = _new_id()
        with self._lock:
            self._conn.execute(
                "INSERT INTO experiments (experiment_id, name, parent_run_id, fork_step_index, "
                "variants_hash, checks_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    experiment_id,
                    name,
                    parent_run_id,
                    fork_step_index,
                    variants_hash,
                    checks_hash,
                    time.time(),
                ),
            )
            self._conn.commit()
        experiment = self.get_experiment(experiment_id)
        if experiment is not None:
            self._emit(
                {
                    "type": "experiment_started",
                    "experiment": experiment_payload(experiment),
                }
            )
        return experiment_id

    def get_experiment(self, experiment_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()

    def experiment_variants(self, experiment_id: str) -> list[dict[str, Any]]:
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            return []
        payload = self._blobs.get_json(str(experiment["variants_hash"]))
        if not isinstance(payload, Mapping):
            return []
        variants = payload.get("variants")
        if not isinstance(variants, list):
            return []
        return [dict(item) for item in variants if isinstance(item, Mapping)]

    def experiment_variant(
        self, experiment_id: str, variant_name: str
    ) -> dict[str, Any] | None:
        for variant in self.experiment_variants(experiment_id):
            if variant.get("name") == variant_name:
                return variant
        return None

    def experiment_runs(self, experiment_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM runs WHERE experiment_id = ? ORDER BY created_at", (experiment_id,)
        ).fetchall()

    def experiment_checks(self, experiment_id: str) -> list[dict[str, Any]]:
        experiment = self.get_experiment(experiment_id)
        if experiment is None or experiment["checks_hash"] is None:
            return []
        payload = self._blobs.get_json(str(experiment["checks_hash"]))
        if not isinstance(payload, Mapping):
            return []
        checks = payload.get("checks")
        if not isinstance(checks, list):
            return []
        return [dict(item) for item in checks if isinstance(item, Mapping)]

    def record_verdict(
        self,
        run_id: str,
        metric_name: str,
        passed: bool,
        reasoning: Mapping[str, Any],
        tier: str = "TIER1",
        step_index: int | None = None,
    ) -> str:
        verdict_id = _new_id()
        with self._lock:
            self._conn.execute(
                "INSERT INTO verdicts (verdict_id, run_id, compared_run_id, step_index, tier, "
                "metric_name, passed, score, position_consistent, reasoning_json, judge_model, "
                "created_at) VALUES (?, ?, NULL, ?, ?, ?, ?, NULL, NULL, ?, NULL, ?)",
                (
                    verdict_id,
                    run_id,
                    step_index,
                    tier,
                    metric_name,
                    1 if passed else 0,
                    canonical_json(dict(reasoning)).decode("utf-8"),
                    time.time(),
                ),
            )
            self._conn.commit()
        return verdict_id

    def verdicts_for_run(self, run_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM verdicts WHERE run_id = ? ORDER BY created_at, metric_name", (run_id,)
        ).fetchall()

    def steps_for(self, run_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY step_index", (run_id,)
        ).fetchall()

    def get_step(self, step_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM steps WHERE step_id = ?", (step_id,)
        ).fetchone()

    def list_runs(
        self,
        limit: int = 100,
        parent_run_id: str | None = None,
        experiment_id: str | None = None,
    ) -> list[sqlite3.Row]:
        sql = (
            "SELECT r.*, (SELECT COUNT(*) FROM steps s WHERE s.run_id = r.run_id) "
            "AS step_count FROM runs r"
        )
        clauses: list[str] = []
        params: list[Any] = []
        if parent_run_id is not None:
            clauses.append("r.parent_run_id = ?")
            params.append(parent_run_id)
        if experiment_id is not None:
            clauses.append("r.experiment_id = ?")
            params.append(experiment_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY r.created_at DESC, r.run_id DESC LIMIT ?"
        params.append(limit)
        return self._conn.execute(sql, params).fetchall()

    def children_of(self, run_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT r.*, (SELECT COUNT(*) FROM steps s WHERE s.run_id = r.run_id) "
            "AS step_count FROM runs r WHERE r.parent_run_id = ? ORDER BY r.created_at",
            (run_id,),
        ).fetchall()

    def list_experiments(self, limit: int = 100) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT e.*, (SELECT COUNT(*) FROM runs r WHERE r.experiment_id = "
            "e.experiment_id) AS run_count FROM experiments e "
            "ORDER BY e.created_at DESC, e.experiment_id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def record_llm_step(
        self,
        run_id: str,
        name: str,
        request_body: Any,
        response_body: Any,
        latency_ms: float,
        error: str | None = None,
        frozen_timestamp: str | None = None,
        source: str = "RECORDED",
    ) -> StepInfo:
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
            frozen_timestamp=frozen_timestamp,
            source=source,
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
        frozen_timestamp: str | None = None,
        source: str = "RECORDED",
    ) -> StepInfo:
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
            frozen_timestamp=frozen_timestamp,
            source=source,
        )

    def record_checkpoint(
        self,
        run_id: str,
        name: str,
        state: Any,
        frozen_timestamp: str | None = None,
        source: str = "RECORDED",
    ) -> StepInfo:
        mk = match_key({"checkpoint": name}, self._masks)
        return self.record_step(
            run_id=run_id,
            step_type="CHECKPOINT",
            name=name,
            match_key_value=mk,
            input_obj={},
            output_obj=state,
            post_state_obj=state,
            frozen_timestamp=frozen_timestamp,
            source=source,
        )

    def record_replayed_step(
        self,
        run_id: str,
        step_type: str,
        name: str,
        input_obj: Any,
        output_obj: Any,
        lookup: LookupResult,
        tool_mode: str | None = None,
        reuse_parent_index: bool = True,
    ) -> StepInfo:
        """Persist a cassette-served step with the parent's time and entropy.

        Steps before a fork point keep the parent's ``step_index`` so checks
        and traces line up. A stub served after the fork point gets the child's
        next index instead.
        """
        if lookup.step_index is None or lookup.frozen_timestamp is None:
            raise ValueError("replayed step requires a matched lookup")
        if step_type == "LLM_CALL":
            mk = match_key(input_obj, self._masks)
        elif step_type == "TOOL_CALL":
            mk = match_key({"tool": name, "input": input_obj}, self._masks)
        elif step_type == "CHECKPOINT":
            mk = match_key({"checkpoint": name}, self._masks)
        else:
            raise ValueError(f"unknown step type: {step_type}")
        return self._record_step_locked(
            run_id=run_id,
            step_type=step_type,
            name=name,
            match_key_value=mk,
            input_obj=input_obj,
            output_obj=output_obj,
            post_state_obj=output_obj if step_type == "CHECKPOINT" else None,
            tool_mode=tool_mode,
            error=lookup.error,
            latency_ms=0.0,
            source="REPLAYED",
            frozen_timestamp=lookup.frozen_timestamp,
            rng_seed=lookup.rng_seed,
            step_index=lookup.step_index if reuse_parent_index else None,
            occurrence=lookup.occurrence,
        )

    def lookup_llm(self, run_id: str, name: str, request_body: Any) -> LookupResult:
        mk = match_key(request_body, self._masks)
        return self._lookup(run_id, "LLM_CALL", name, mk, request_body)

    def lookup_tool(self, run_id: str, name: str, input_obj: Any) -> LookupResult:
        mk = match_key({"tool": name, "input": input_obj}, self._masks)
        return self._lookup(run_id, "TOOL_CALL", name, mk, input_obj)

    def lookup_checkpoint(self, run_id: str, name: str) -> LookupResult:
        mk = match_key({"checkpoint": name}, self._masks)
        return self._lookup(run_id, "CHECKPOINT", name, mk, {})

    def _lookup(
        self, run_id: str, step_type: str, name: str, match_key_value: str, incoming_input: Any
    ) -> LookupResult:
        with self._lock:
            run = self.get_run(run_id)
            if run is None:
                raise KeyError(f"unknown run: {run_id}")
            parent_run_id = run["parent_run_id"]
            occurrence = int(
                self._conn.execute(
                    "SELECT COUNT(*) AS n FROM steps WHERE run_id = ? AND name = ?",
                    (run_id, name),
                ).fetchone()["n"]
            )
            if parent_run_id is None:
                return LookupResult(
                    found=False, matched=False, occurrence=occurrence, step=None
                )
            row = find_recorded_step(self._conn, str(parent_run_id), name, occurrence)
            if row is None:
                return LookupResult(
                    found=False, matched=False, occurrence=occurrence, step=None
                )
            matched = str(row["match_key"]) == match_key_value
            diff = None
            if not matched:
                try:
                    recorded_input = self._blobs.get_json(str(row["input_hash"]))
                except KeyError:
                    recorded_input = None
                diff = normalized_diff(
                    recorded_input,
                    incoming_input,
                    self._masks,
                    step_label(step_type, name, occurrence),
                )
            return LookupResult(
                found=True,
                matched=matched,
                occurrence=occurrence,
                step=row,
                diff=diff,
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
        frozen_timestamp: str | None = None,
        source: str = "RECORDED",
    ) -> StepInfo:
        return self._record_step_locked(
            run_id=run_id,
            step_type=step_type,
            name=name,
            match_key_value=match_key_value,
            input_obj=input_obj,
            output_obj=output_obj,
            post_state_obj=post_state_obj,
            tool_mode=tool_mode,
            error=error,
            latency_ms=latency_ms,
            source=source,
            frozen_timestamp=frozen_timestamp,
            rng_seed=None,
            step_index=None,
            occurrence=None,
        )

    def _record_step_locked(
        self,
        *,
        run_id: str,
        step_type: str,
        name: str,
        match_key_value: str,
        input_obj: Any,
        output_obj: Any,
        post_state_obj: Any,
        tool_mode: str | None,
        error: str | None,
        latency_ms: float,
        source: str,
        frozen_timestamp: str | None,
        rng_seed: int | None,
        step_index: int | None,
        occurrence: int | None,
    ) -> StepInfo:
        with self._lock:
            run = self.get_run(run_id)
            if run is None:
                raise KeyError(f"unknown run: {run_id}")
            if step_index is None:
                step_index = self._next_index.get(run_id)
                if step_index is None:
                    row = self._conn.execute(
                        "SELECT COALESCE(MAX(step_index) + 1, 0) AS next FROM steps "
                        "WHERE run_id = ?",
                        (run_id,),
                    ).fetchone()
                    step_index = int(row["next"])
            if occurrence is None:
                occurrence = int(
                    self._conn.execute(
                        "SELECT COUNT(*) AS n FROM steps WHERE run_id = ? AND name = ?",
                        (run_id, name),
                    ).fetchone()["n"]
                )
            timestamp = frozen_timestamp or real_now_iso()
            seed = rng_seed if rng_seed is not None else boundary_seed(
                int(run["run_seed"]), step_index
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
            step_id = _new_id()
            redacted_error = redact_text(error, self._redaction) if error else None
            self._conn.execute(
                _INSERT_STEP_SQL,
                (
                    step_id,
                    run_id,
                    step_index,
                    step_type,
                    name,
                    occurrence,
                    match_key_value,
                    input_hash,
                    output_hash,
                    post_state_hash,
                    timestamp,
                    seed,
                    tool_mode,
                    source,
                    redacted_error,
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
            self._emit(
                {
                    "type": "step_added",
                    "run_id": run_id,
                    "step": {
                        "step_id": step_id,
                        "run_id": run_id,
                        "step_index": step_index,
                        "step_type": step_type,
                        "name": name,
                        "occurrence": occurrence,
                        "source": source,
                        "tool_mode": tool_mode,
                        "error": redacted_error,
                        "latency_ms": latency_ms,
                        "frozen_timestamp": timestamp,
                        "rng_seed": seed,
                    },
                }
            )
            return StepInfo(
                step_index=step_index,
                occurrence=occurrence,
                frozen_timestamp=timestamp,
                rng_seed=seed,
            )
