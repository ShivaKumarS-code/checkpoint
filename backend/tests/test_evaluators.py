"""Tests for Tier 1 deterministic checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from checkpoint.daemon.recorder import Recorder
from checkpoint.evaluators.deterministic import (
    CheckResult,
    CheckSpec,
    run_checks,
    validate_checks,
)
from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import init_db


def _make_run(tmp_path: Path) -> tuple[Path, str]:
    db_path = tmp_path / "cassettes.db"
    conn = init_db(db_path)
    try:
        recorder = Recorder(conn, BlobStore(conn))
        created = recorder.create_run("tier1", "RECORD")
        run_id = str(created["run_id"])
        recorder.record_llm_step(
            run_id,
            "plan",
            {"messages": [{"role": "user", "content": "plan"}]},
            {"choices": [{"message": {"content": "ok"}}]},
            latency_ms=1.0,
        )
        recorder.record_tool_step(
            run_id,
            "lookup_order",
            {"args": ["ORD-1"], "kwargs": {}},
            {"order_id": "ORD-1"},
        )
        recorder.record_tool_step(
            run_id,
            "issue_refund",
            {"args": ["ORD-1", 10.0], "kwargs": {}},
            None,
            tool_mode="stub",
            error="RuntimeError: boom",
        )
        recorder.record_checkpoint(
            run_id,
            "after_refund",
            {"refund": None, "nested": {"x": 1}, "reply": "cannot approve a refund"},
        )
        recorder.finish_run(run_id, "COMPLETED")
        return db_path, run_id
    finally:
        conn.close()


def _checks(
    db_path: Path, run_id: str, checks: list[CheckSpec], base: Path | None = None
) -> list[CheckResult]:
    conn = init_db(db_path)
    try:
        return run_checks(conn, BlobStore(conn), run_id, checks, base)
    finally:
        conn.close()


def test_no_errors_passes_without_failures_and_fails_on_step_errors(tmp_path: Path) -> None:
    db_path, run_id = _make_run(tmp_path)
    results = _checks(db_path, run_id, [CheckSpec(type="no_errors")])
    assert results[0].passed is False
    assert results[0].reasoning["errors"][0]["name"] == "issue_refund"

    conn = init_db(db_path)
    try:
        recorder = Recorder(conn, BlobStore(conn))
        clean = recorder.create_run("clean", "RECORD")
        recorder.record_tool_step(str(clean["run_id"]), "lookup", {}, {"ok": True})
        recorder.finish_run(str(clean["run_id"]), "COMPLETED")
    finally:
        conn.close()
    clean_results = _checks(db_path, str(clean["run_id"]), [CheckSpec(type="no_errors")])
    assert clean_results[0].passed is True


def test_tool_checks_and_order(tmp_path: Path) -> None:
    db_path, run_id = _make_run(tmp_path)
    results = _checks(
        db_path,
        run_id,
        [
            CheckSpec(type="tool_called", name="lookup_order"),
            CheckSpec(type="tool_not_called", name="delete_order"),
            CheckSpec(type="tool_not_called", name="issue_refund"),
            CheckSpec(type="tool_order", names=["lookup_order", "issue_refund"]),
            CheckSpec(type="tool_order", names=["issue_refund", "lookup_order"]),
        ],
    )
    assert results[0].passed is True
    assert results[1].passed is True
    assert results[2].passed is False
    assert results[3].passed is True
    assert results[4].passed is False


def test_state_equals_resolves_paths(tmp_path: Path) -> None:
    db_path, run_id = _make_run(tmp_path)
    results = _checks(
        db_path,
        run_id,
        [
            CheckSpec(type="state_equals", checkpoint="after_refund", path="refund", value=None),
            CheckSpec(type="state_equals", checkpoint="after_refund", path="nested.x", value=1),
            CheckSpec(type="state_equals", checkpoint="after_refund", path="nested.x", value=2),
            CheckSpec(type="state_equals", checkpoint="missing", path="x", value=1),
        ],
    )
    assert results[0].passed is True
    assert results[1].passed is True
    assert results[2].passed is False
    assert results[2].reasoning["actual"] == 1
    assert results[3].passed is False


def test_final_output_regex_and_schema(tmp_path: Path) -> None:
    db_path, run_id = _make_run(tmp_path)
    results = _checks(
        db_path,
        run_id,
        [
            CheckSpec(type="final_output_matches", regex="cannot approve"),
            CheckSpec(type="final_output_matches", regex="refund issued"),
            CheckSpec(
                type="final_output_matches",
                schema={
                    "type": "object",
                    "required": ["refund"],
                    "properties": {"nested": {"type": "object"}},
                },
            ),
            CheckSpec(
                type="final_output_matches",
                schema={"type": "object", "required": ["missing_key"]},
            ),
        ],
    )
    assert results[0].passed is True
    assert results[1].passed is False
    assert results[2].passed is True
    assert results[3].passed is False


def test_step_count_lte(tmp_path: Path) -> None:
    db_path, run_id = _make_run(tmp_path)
    results = _checks(
        db_path,
        run_id,
        [
            CheckSpec(type="step_count_lte", max_steps=4),
            CheckSpec(type="step_count_lte", max_steps=3),
        ],
    )
    assert results[0].passed is True
    assert results[1].passed is False


def test_python_check_receives_steps_and_supports_reasoning(tmp_path: Path) -> None:
    db_path, run_id = _make_run(tmp_path)
    checks_dir = tmp_path / "checks"
    checks_dir.mkdir()
    (checks_dir / "my_checks.py").write_text(
        "def has_refund_error(steps):\n"
        "    return any(step['name'] == 'issue_refund' and step['error'] for step in steps)\n"
        "def failing(steps):\n"
        "    return False, {'why': 'always fails', 'count': len(steps)}\n",
        encoding="utf-8",
    )
    results = _checks(
        db_path,
        run_id,
        [
            CheckSpec(type="python", path="checks/my_checks.py::has_refund_error"),
            CheckSpec(type="python", path="checks/my_checks.py::failing"),
            CheckSpec(type="python", path="checks/my_checks.py::nope"),
        ],
        tmp_path,
    )
    assert results[0].passed is True
    assert results[1].passed is False
    assert results[1].reasoning["detail"]["why"] == "always fails"
    assert results[2].passed is False
    assert "not found" in results[2].reasoning["error"]


def test_check_validation_rejects_incomplete_checks() -> None:
    with pytest.raises(ValueError):
        validate_checks([CheckSpec(type="tool_called")])
    with pytest.raises(ValueError):
        validate_checks([CheckSpec(type="tool_order")])
    with pytest.raises(ValueError):
        validate_checks([CheckSpec(type="final_output_matches")])
    with pytest.raises(ValueError):
        validate_checks([CheckSpec(type="final_output_matches", regex="[")])
    with pytest.raises(ValueError):
        validate_checks([CheckSpec(type="state_equals", checkpoint="c")])
    with pytest.raises(ValueError):
        validate_checks([CheckSpec(type="step_count_lte")])
    with pytest.raises(ValueError):
        validate_checks([CheckSpec(type="python", path="no_function")])
