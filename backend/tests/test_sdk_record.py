"""Tests for the SDK recording path: @tool, @step and checkpoint()."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import httpx
import pytest

import checkpoint
from checkpoint.sdk import client as sdk_client
from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import connect

from .conftest import DaemonHandle


def bind(monkeypatch: pytest.MonkeyPatch, daemon: DaemonHandle, run_id: str | None = None) -> None:
    monkeypatch.setenv("CHECKPOINT_DAEMON_URL", daemon.url)
    monkeypatch.setenv("CHECKPOINT_MODE", "record")
    if run_id is None:
        monkeypatch.delenv("CHECKPOINT_RUN_ID", raising=False)
    else:
        monkeypatch.setenv("CHECKPOINT_RUN_ID", run_id)


def fetch_steps(db_path: Path) -> list[sqlite3.Row]:
    conn = connect(db_path)
    try:
        return conn.execute("SELECT * FROM steps ORDER BY step_index").fetchall()
    finally:
        conn.close()


def test_init_registers_a_run_when_no_run_id_is_given(
    daemon: DaemonHandle, monkeypatch: pytest.MonkeyPatch
) -> None:
    bind(monkeypatch, daemon)
    run_id = checkpoint.init(run_name="sdk-test")
    assert run_id
    conn = connect(daemon.db_path)
    try:
        run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        assert run["name"] == "sdk-test"
        assert run["mode"] == "RECORD"
        assert run["status"] == "RUNNING"
    finally:
        conn.close()


def test_tool_records_inputs_outputs_and_mode(
    daemon: DaemonHandle, monkeypatch: pytest.MonkeyPatch
) -> None:
    bind(monkeypatch, daemon)
    checkpoint.init(run_name="sdk-test")

    @checkpoint.tool(mode="pure")
    def add(a: int, b: int = 1) -> int:
        return a + b

    assert add(2, b=3) == 5
    steps = fetch_steps(daemon.db_path)
    assert len(steps) == 1
    step = steps[0]
    assert step["step_type"] == "TOOL_CALL"
    assert step["name"] == "add"
    assert step["tool_mode"] == "pure"
    assert step["latency_ms"] >= 0.0
    conn = connect(daemon.db_path)
    try:
        blobs = BlobStore(conn)
        assert blobs.get_json(step["input_hash"]) == {"args": [2], "kwargs": {"b": 3}}
        assert blobs.get_json(step["output_hash"]) == 5
    finally:
        conn.close()


def test_tool_error_is_recorded_and_reraised(
    daemon: DaemonHandle, monkeypatch: pytest.MonkeyPatch
) -> None:
    bind(monkeypatch, daemon)
    checkpoint.init(run_name="sdk-test")

    @checkpoint.tool(mode="pure")
    def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        boom()
    step = fetch_steps(daemon.db_path)[0]
    assert step["output_hash"] is None
    assert "ValueError: nope" in step["error"]


def test_stub_tool_records_its_mode(daemon: DaemonHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    bind(monkeypatch, daemon)
    checkpoint.init(run_name="sdk-test")

    @checkpoint.tool(mode="stub")
    def side_effect(x: int) -> int:
        return x

    side_effect(1)
    step = fetch_steps(daemon.db_path)[0]
    assert step["tool_mode"] == "stub"


def test_checkpoint_records_state_snapshot(
    daemon: DaemonHandle, monkeypatch: pytest.MonkeyPatch
) -> None:
    bind(monkeypatch, daemon)
    checkpoint.init(run_name="sdk-test")
    state = {"order": {"status": "refunded"}, "amount": 84.5}
    checkpoint.checkpoint("after_refund", state)

    step = fetch_steps(daemon.db_path)[0]
    assert step["step_type"] == "CHECKPOINT"
    assert step["name"] == "after_refund"
    assert step["post_state_hash"] == step["output_hash"]
    conn = connect(daemon.db_path)
    try:
        blobs = BlobStore(conn)
        assert blobs.get_json(step["output_hash"]) == state
        assert blobs.get_json(step["input_hash"]) == {}
    finally:
        conn.close()


def test_step_labels_llm_calls_through_the_proxy(
    daemon: DaemonHandle, monkeypatch: pytest.MonkeyPatch
) -> None:
    bind(monkeypatch, daemon)
    checkpoint.init(run_name="sdk-test")

    @checkpoint.step(name="plan")
    def call_llm() -> httpx.Response:
        return httpx.post(
            f"{daemon.url}/v1/chat/completions",
            json={"model": "demo-model", "messages": [{"role": "user", "content": "hi"}]},
            timeout=10.0,
        )

    response = call_llm()
    assert response.status_code == 200
    step = fetch_steps(daemon.db_path)[0]
    assert step["step_type"] == "LLM_CALL"
    assert step["name"] == "plan"


def test_groq_style_path_is_tagged_by_the_sdk(
    daemon: DaemonHandle, monkeypatch: pytest.MonkeyPatch
) -> None:
    bind(monkeypatch, daemon)
    checkpoint.init(run_name="sdk-test")

    @checkpoint.step(name="review")
    def call_groq() -> httpx.Response:
        return httpx.post(
            f"{daemon.url}/openai/v1/chat/completions",
            json={"model": "demo-model", "messages": [{"role": "user", "content": "hi"}]},
            timeout=10.0,
        )

    response = call_groq()
    assert response.status_code == 200
    step = fetch_steps(daemon.db_path)[0]
    assert (step["step_type"], step["name"], step["source"]) == (
        "LLM_CALL",
        "review",
        "RECORDED",
    )


def test_everything_is_a_no_op_without_a_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHECKPOINT_DAEMON_URL", raising=False)
    sdk_client.reset()
    assert checkpoint.init(run_name="standalone") is None

    @checkpoint.tool(mode="pure")
    def double(x: int) -> int:
        return x * 2

    assert double(21) == 42
    checkpoint.checkpoint("later", {"a": 1})
