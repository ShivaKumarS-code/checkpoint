"""JSON-ready views of stored rows shared by events and the query API."""

from __future__ import annotations

import sqlite3
from typing import Any

RunRow = sqlite3.Row | dict[str, Any]

RUN_FIELDS = (
    "run_id",
    "name",
    "mode",
    "status",
    "parent_run_id",
    "experiment_id",
    "variant_name",
    "forked_at_step",
    "run_seed",
    "created_at",
    "total_tokens",
    "total_latency_ms",
    "divergence_step",
)

STEP_FIELDS = (
    "step_id",
    "run_id",
    "step_index",
    "step_type",
    "name",
    "occurrence",
    "source",
    "tool_mode",
    "error",
    "latency_ms",
    "frozen_timestamp",
    "rng_seed",
)

EXPERIMENT_FIELDS = (
    "experiment_id",
    "name",
    "parent_run_id",
    "fork_step_index",
    "created_at",
)


def _contains(row: RunRow, key: str) -> bool:
    if isinstance(row, dict):
        return key in row
    try:
        return key in row.keys()
    except AttributeError:
        return False


def run_payload(row: RunRow) -> dict[str, Any]:
    payload = {field: row[field] for field in RUN_FIELDS}
    if _contains(row, "step_count"):
        payload["step_count"] = int(row["step_count"])
    return payload


def step_payload(row: RunRow) -> dict[str, Any]:
    payload = {field: row[field] for field in STEP_FIELDS}
    for optional in ("input_hash", "output_hash", "post_state_hash", "match_key"):
        if _contains(row, optional):
            payload[optional] = row[optional]
    return payload


def experiment_payload(row: RunRow) -> dict[str, Any]:
    payload = {field: row[field] for field in EXPERIMENT_FIELDS}
    if _contains(row, "run_count"):
        payload["run_count"] = int(row["run_count"])
    if _contains(row, "semantic_matches"):
        count = int(row["semantic_matches"])
        noun = "step" if count == 1 else "steps"
        payload["semantic_matches"] = count
        payload["semantic_summary"] = f"{count} {noun} served by semantic match"
    return payload
