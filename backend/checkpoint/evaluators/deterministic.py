"""Tier 1 deterministic checks (SPEC.md section 9).

Every check receives a materialized view of the child run: the run row, its
steps in order, and resolved input/output blobs. Checks return a verdict with a
JSON-able reasoning payload; they never call an LLM.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import msgspec

from checkpoint.matching.normalize import canonical_json
from checkpoint.storage.cas import BlobStore

CheckType = Literal[
    "no_errors",
    "tool_called",
    "tool_not_called",
    "tool_order",
    "final_output_matches",
    "state_equals",
    "step_count_lte",
    "python",
    "rubric",
]


class CheckSpec(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    """One check from variants.yaml (Tier 1 deterministic or Tier 2 rubric)."""

    type: CheckType
    required: bool = True
    name: str | None = None
    names: list[str] | None = None
    regex: str | None = None
    schema: dict[str, Any] | None = None
    checkpoint: str | None = None
    path: str | None = None
    value: Any = None
    max_steps: int | None = None
    rubric: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class CheckResult:
    check_type: str
    label: str
    passed: bool
    required: bool
    reasoning: dict[str, Any]
    tier: str = "TIER1"
    score: float | None = None
    judge_model: str | None = None


def validate_checks(checks: Sequence[CheckSpec]) -> None:
    """Raise ValueError for a check that cannot be evaluated."""
    for check in checks:
        if check.type in ("tool_called", "tool_not_called") and not check.name:
            raise ValueError(f"{check.type} requires name")
        if check.type == "tool_order" and not check.names:
            raise ValueError("tool_order requires names")
        if check.type == "final_output_matches":
            if check.regex is None and check.schema is None:
                raise ValueError("final_output_matches requires regex or schema")
            if check.regex is not None:
                try:
                    re.compile(check.regex)
                except re.error as exc:
                    raise ValueError(f"invalid regex: {exc}") from exc
        if check.type == "state_equals":
            if not check.checkpoint:
                raise ValueError("state_equals requires checkpoint")
            if not check.path:
                raise ValueError("state_equals requires path")
        if check.type == "step_count_lte" and check.max_steps is None:
            raise ValueError("step_count_lte requires max_steps")
        if check.type == "python":
            if not check.path or "::" not in check.path:
                raise ValueError("python checks require path: file.py::function")
        if check.type == "rubric" and not (check.rubric or "").strip():
            raise ValueError("rubric checks require a non-empty rubric")


def check_label(check: CheckSpec) -> str:
    if check.type in ("tool_called", "tool_not_called") and check.name:
        return f"{check.type}:{check.name}"
    if check.type == "state_equals" and check.checkpoint and check.path:
        return f"state_equals:{check.checkpoint}:{check.path}"
    if check.type == "rubric" and check.name:
        return f"rubric:{check.name}"
    if check.type == "final_output_matches":
        return "final_output_matches"
    return check.type


def run_checks(
    conn: sqlite3.Connection,
    blob_store: BlobStore,
    run_id: str,
    checks: Sequence[CheckSpec],
    base_dir: str | Path | None = None,
    *,
    judge: Any = None,
    judge_model: str = "gpt-4o-mini",
) -> list[CheckResult]:
    """Run deterministic checks and Tier 2 rubric checks in order."""
    from checkpoint.evaluators.rubric import evaluate_rubric

    view = load_run_view(conn, blob_store, run_id)
    results: list[CheckResult] = []
    for check in checks:
        if check.type == "rubric":
            results.append(
                evaluate_rubric(
                    conn,
                    view,
                    check,
                    judge,
                    default_model=judge_model,
                )
            )
        else:
            results.append(evaluate(check, view, base_dir))
    return results


def load_run_view(
    conn: sqlite3.Connection, blob_store: BlobStore, run_id: str
) -> dict[str, Any]:
    run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if run is None:
        raise KeyError(f"unknown run: {run_id}")
    rows = conn.execute(
        "SELECT * FROM steps WHERE run_id = ? ORDER BY step_index", (run_id,)
    ).fetchall()
    steps: list[dict[str, Any]] = []
    for row in rows:
        output: Any = None
        if row["output_hash"] is not None:
            output = blob_store.get_json(str(row["output_hash"]))
        steps.append(
            {
                "step_id": row["step_id"],
                "step_index": int(row["step_index"]),
                "step_type": row["step_type"],
                "name": row["name"],
                "occurrence": int(row["occurrence"]),
                "source": row["source"],
                "tool_mode": row["tool_mode"],
                "error": row["error"],
                "latency_ms": float(row["latency_ms"]),
                "input": blob_store.get_json(str(row["input_hash"])),
                "output": output,
            }
        )
    return {
        "run": dict(run),
        "steps": steps,
        "final_output": _final_output(steps),
    }


def _final_output(steps: Sequence[Mapping[str, Any]]) -> Any:
    for step in reversed(steps):
        if step["step_type"] == "CHECKPOINT":
            return step["output"]
    for step in reversed(steps):
        if step["output"] is not None:
            return step["output"]
    return None


def evaluate(
    check: CheckSpec, view: Mapping[str, Any], base_dir: str | Path | None = None
) -> CheckResult:
    label = check_label(check)
    try:
        passed, reasoning = _evaluate(check, view, base_dir)
    except Exception as exc:
        passed, reasoning = False, {"error": f"{type(exc).__name__}: {exc}"}
    return CheckResult(
        check_type=check.type,
        label=label,
        passed=passed,
        required=check.required,
        reasoning=reasoning,
    )


def _evaluate(
    check: CheckSpec, view: Mapping[str, Any], base_dir: str | Path | None
) -> tuple[bool, dict[str, Any]]:
    run: Mapping[str, Any] = view["run"]
    steps: list[Mapping[str, Any]] = view["steps"]

    if check.type == "no_errors":
        errors = [
            {"step_index": step["step_index"], "name": step["name"], "error": step["error"]}
            for step in steps
            if step["error"]
        ]
        status_ok = run["status"] == "COMPLETED"
        return status_ok and not errors, {
            "status": run["status"],
            "divergence_step": run["divergence_step"],
            "errors": errors,
        }

    if check.type == "tool_called":
        found = any(
            step["step_type"] == "TOOL_CALL" and step["name"] == check.name for step in steps
        )
        return found, {"name": check.name, "found": found}

    if check.type == "tool_not_called":
        calls = [
            step["step_index"]
            for step in steps
            if step["step_type"] == "TOOL_CALL" and step["name"] == check.name
        ]
        return not calls, {"name": check.name, "called_at": calls}

    if check.type == "tool_order":
        calls = [
            str(step["name"]) for step in steps if step["step_type"] == "TOOL_CALL"
        ]
        wanted = list(check.names or [])
        position = 0
        for name in calls:
            if position < len(wanted) and name == wanted[position]:
                position += 1
        return position == len(wanted), {"wanted": wanted, "observed": calls}

    if check.type == "final_output_matches":
        output = view["final_output"]
        reasons: dict[str, Any] = {"output": output}
        if check.regex is not None:
            text = output if isinstance(output, str) else canonical_json(output).decode("utf-8")
            matched = re.search(check.regex, text) is not None
            reasons["regex_matched"] = matched
            if not matched:
                return False, reasons
        if check.schema is not None:
            error = _schema_error(output, check.schema)
            reasons["schema_error"] = error
            if error is not None:
                return False, reasons
        return True, reasons

    if check.type == "state_equals":
        state = _checkpoint_after(steps, str(check.checkpoint))
        if state is _MISSING:
            return False, {"error": f"checkpoint not found: {check.checkpoint}"}
        try:
            actual = _resolve_path(state, str(check.path))
        except KeyError as exc:
            return False, {"error": str(exc), "path": check.path}
        return actual == check.value, {
            "checkpoint": check.checkpoint,
            "path": check.path,
            "expected": check.value,
            "actual": actual,
        }

    if check.type == "step_count_lte":
        count = len(steps)
        limit = int(check.max_steps or 0)
        return count <= limit, {"count": count, "max_steps": limit}

    if check.type == "python":
        return _run_python_check(str(check.path), steps, base_dir)

    return False, {"error": f"unknown check type: {check.type}"}


def _checkpoint_after(steps: Sequence[Mapping[str, Any]], name: str) -> Any:
    matches = [step for step in steps if step["step_type"] == "CHECKPOINT" and step["name"] == name]
    if not matches:
        return _MISSING
    return matches[-1]["output"]


def _resolve_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                raise KeyError(f"path not found: {path} (missing {part})")
            current = current[part]
        elif isinstance(current, list):
            try:
                index = int(part)
            except ValueError as exc:
                raise KeyError(f"path not found: {path} (not a list index: {part})") from exc
            if index < 0 or index >= len(current):
                raise KeyError(f"path not found: {path} (index {index} out of range)")
            current = current[index]
        else:
            raise KeyError(f"path not found: {path} (cannot descend into {type(current).__name__})")
    return current


class _Missing:
    pass


_MISSING = _Missing()


def _schema_error(value: Any, schema: Mapping[str, Any]) -> str | None:
    """Validate the small JSON-schema subset used by Tier 1 checks."""
    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        if not any(_matches_type(value, str(item)) for item in types):
            return f"expected type {expected}, got {_type_name(value)}"
    if "enum" in schema and value not in schema["enum"]:
        return f"value {value!r} is not one of {schema['enum']!r}"
    if isinstance(value, Mapping):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                return f"missing required property: {key}"
        properties = schema.get("properties", {})
        if isinstance(properties, Mapping):
            for key, child in properties.items():
                if key in value and isinstance(child, Mapping):
                    error = _schema_error(value[key], child)
                    if error is not None:
                        return f"{key}: {error}"
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                error = _schema_error(item, items)
                if error is not None:
                    return f"[{index}]: {error}"
    return None


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _run_python_check(
    spec: str, steps: list[Mapping[str, Any]], base_dir: str | Path | None
) -> tuple[bool, dict[str, Any]]:
    file_name, _, function_name = spec.partition("::")
    path = Path(file_name)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    path = path.resolve()
    if not path.is_file():
        return False, {"error": f"check file not found: {path}"}
    module_name = f"checkpoint_check_{abs(hash((str(path), function_name)))}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        return False, {"error": f"cannot load check file: {path}"}
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    function = getattr(module, function_name, None)
    if not callable(function):
        return False, {"error": f"check function not found: {function_name}"}
    result = function(steps)
    if isinstance(result, tuple):
        passed = bool(result[0])
        detail = result[1] if len(result) > 1 else None
        return passed, {"result": passed, "detail": _jsonable(detail)}
    return bool(result), {"result": bool(result)}


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value
