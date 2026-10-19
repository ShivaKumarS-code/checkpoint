"""Tests for replay lookup, divergence, live misses and replayed step records."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import httpx

from checkpoint.config import Config, ReplayConfig
from checkpoint.daemon.app import run_daemon
from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import connect
from examples import fake_llm

from .conftest import DaemonHandle
from .test_proxy_record import chat_body, create_run, fetch_steps, post_chat


def run_row(db_path: Path, run_id: str) -> sqlite3.Row:
    conn = connect(db_path)
    try:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        assert row is not None
        return row
    finally:
        conn.close()


def child_of(daemon: DaemonHandle, parent_run_id: str) -> str:
    parent = run_row(daemon.db_path, parent_run_id)
    return create_run(
        daemon.url,
        name=str(parent["name"]),
        mode="REPLAY",
        parent_run_id=parent_run_id,
        run_seed=int(parent["run_seed"]),
    )


def steps_for(db_path: Path, run_id: str) -> list[sqlite3.Row]:
    conn = connect(db_path)
    try:
        return conn.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY step_index", (run_id,)
        ).fetchall()
    finally:
        conn.close()


def post_step(url: str, payload: dict[str, object]) -> httpx.Response:
    return httpx.post(f"{url}/api/sdk/step", json=payload, timeout=10.0)


def test_replay_serves_the_recorded_llm_output(
    daemon: DaemonHandle, fake_llm_server: fake_llm.FakeLLMServer
) -> None:
    fake_llm.reset()
    parent = create_run(daemon.url)
    body = chat_body("replay me")
    recorded = post_chat(daemon.url, body, run_id=parent, step_name="plan")
    assert recorded.status_code == 200
    parent_step = fetch_steps(daemon.db_path)[0]

    fake_llm.reset()
    child = child_of(daemon, parent)
    replayed = post_chat(daemon.url, body, run_id=child, step_name="plan")
    assert replayed.status_code == 200
    assert replayed.json() == recorded.json()
    assert fake_llm.call_count() == 0

    child_steps = steps_for(daemon.db_path, child)
    assert len(child_steps) == 1
    step = child_steps[0]
    assert step["source"] == "REPLAYED"
    assert step["step_index"] == parent_step["step_index"]
    assert step["occurrence"] == parent_step["occurrence"]
    assert step["frozen_timestamp"] == parent_step["frozen_timestamp"]
    assert step["rng_seed"] == parent_step["rng_seed"]
    assert step["match_key"] == parent_step["match_key"]
    assert step["latency_ms"] == 0.0
    assert replayed.headers["x-checkpoint-frozen-timestamp"] == parent_step["frozen_timestamp"]
    assert replayed.headers["x-checkpoint-rng-seed"] == str(parent_step["rng_seed"])


def test_replay_matches_by_name_and_occurrence_not_index(daemon: DaemonHandle) -> None:
    parent = create_run(daemon.url)
    post_chat(daemon.url, chat_body("plan this"), run_id=parent, step_name="plan")
    post_step(
        daemon.url,
        {
            "run_id": parent,
            "step_type": "TOOL_CALL",
            "name": "lookup",
            "tool_mode": "pure",
            "input": {"args": [1], "kwargs": {}},
            "output": 1,
        },
    )
    post_chat(daemon.url, chat_body("decide this"), run_id=parent, step_name="decide")

    child = child_of(daemon, parent)
    first = post_chat(daemon.url, chat_body("plan this"), run_id=child, step_name="plan")
    second = post_chat(daemon.url, chat_body("decide this"), run_id=child, step_name="decide")
    assert first.status_code == 200
    assert second.status_code == 200
    names = [(row["name"], row["occurrence"]) for row in steps_for(daemon.db_path, child)]
    assert names == [("plan", 0), ("decide", 0)]


def test_replay_repeats_the_same_name_with_increasing_occurrence(
    daemon: DaemonHandle,
) -> None:
    parent = create_run(daemon.url)
    post_chat(daemon.url, chat_body("first"), run_id=parent, step_name="llm_call")
    post_chat(daemon.url, chat_body("second"), run_id=parent, step_name="llm_call")

    child = child_of(daemon, parent)
    for content in ("first", "second"):
        response = post_chat(daemon.url, chat_body(content), run_id=child, step_name="llm_call")
        assert response.status_code == 200
    assert [(row["occurrence"], row["source"]) for row in steps_for(daemon.db_path, child)] == [
        (0, "REPLAYED"),
        (1, "REPLAYED"),
    ]


def test_changed_request_diverges_with_a_readable_diff(daemon: DaemonHandle) -> None:
    parent = create_run(daemon.url)
    post_chat(daemon.url, chat_body("ORIGINAL ticket"), run_id=parent, step_name="plan")
    child = child_of(daemon, parent)

    response = post_chat(daemon.url, chat_body("CHANGED ticket"), run_id=child, step_name="plan")
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["type"] == "checkpoint_divergence"
    assert 'LLM call "plan" occurrence 0' in error["message"]
    assert "-" in error["diff"] and "+" in error["diff"]
    assert "ORIGINAL ticket" in error["diff"]
    assert "CHANGED ticket" in error["diff"]
    assert steps_for(daemon.db_path, child) == []

    run = run_row(daemon.db_path, child)
    assert run["status"] == "DIVERGED"
    assert run["divergence_step"] == 0
    conn = connect(daemon.db_path)
    try:
        diff = BlobStore(conn).get_json(run["divergence_diff_hash"])["diff"]
        assert "CHANGED ticket" in diff
    finally:
        conn.close()


def test_diverged_run_stays_diverged_when_finished(daemon: DaemonHandle) -> None:
    parent = create_run(daemon.url)
    post_chat(daemon.url, chat_body("one"), run_id=parent, step_name="plan")
    child = child_of(daemon, parent)
    post_chat(daemon.url, chat_body("two"), run_id=child, step_name="plan")

    from checkpoint.daemon.recorder import Recorder

    conn = connect(daemon.db_path)
    try:
        Recorder(conn, BlobStore(conn)).finish_run(child, "COMPLETED")
    finally:
        conn.close()
    assert run_row(daemon.db_path, child)["status"] == "DIVERGED"


def test_strict_miss_without_a_parent_step_is_a_divergence(daemon: DaemonHandle) -> None:
    parent = create_run(daemon.url)
    child = child_of(daemon, parent)
    response = post_chat(daemon.url, chat_body(), run_id=child, step_name="plan")
    assert response.status_code == 409
    assert "no recorded LLM call" in response.json()["error"]["diff"]


def test_live_llm_miss_policy_forwards_and_records_live(
    fake_llm_server: fake_llm.FakeLLMServer, tmp_path: Path
) -> None:
    db_path = tmp_path / "cassettes.db"
    config = Config(replay=ReplayConfig(llm_miss_policy="live"))
    running = run_daemon(db_path, fake_llm_server.url, config=config, port=0)
    try:
        parent = create_run(running.url)
        child = create_run(
            running.url,
            name="live-miss",
            mode="REPLAY",
            parent_run_id=parent,
            run_seed=int(run_row(db_path, parent)["run_seed"]),
        )
        fake_llm.reset()
        response = post_chat(running.url, chat_body("live now"), run_id=child, step_name="plan")
        assert response.status_code == 200
        assert fake_llm.call_count() == 1
        step = steps_for(db_path, child)[0]
        assert step["source"] == "LIVE"
        assert step["error"] is None
    finally:
        running.stop()


def test_tool_replay_and_stub_misses(daemon: DaemonHandle) -> None:
    parent = create_run(daemon.url)
    tool_input = {"args": ["ORD-1", 84.5], "kwargs": {}}
    assert post_step(
        daemon.url,
        {
            "run_id": parent,
            "step_type": "TOOL_CALL",
            "name": "issue_refund",
            "tool_mode": "stub",
            "input": tool_input,
            "output": {"refund_id": "rf_ord_1"},
        },
    ).status_code == 200
    parent_step = steps_for(daemon.db_path, parent)[0]
    child = child_of(daemon, parent)

    hit = httpx.post(
        f"{daemon.url}/api/sdk/lookup",
        json={
            "run_id": child,
            "step_type": "TOOL_CALL",
            "name": "issue_refund",
            "tool_mode": "stub",
            "input": tool_input,
        },
        timeout=10.0,
    )
    assert hit.status_code == 200
    assert hit.json()["status"] == "replayed"
    assert hit.json()["output"] == {"refund_id": "rf_ord_1"}
    child_step = steps_for(daemon.db_path, child)[0]
    assert child_step["source"] == "REPLAYED"
    assert child_step["tool_mode"] == "stub"
    assert child_step["frozen_timestamp"] == parent_step["frozen_timestamp"]

    changed_child = child_of(daemon, parent)
    changed = httpx.post(
        f"{daemon.url}/api/sdk/lookup",
        json={
            "run_id": changed_child,
            "step_type": "TOOL_CALL",
            "name": "issue_refund",
            "tool_mode": "stub",
            "input": {"args": ["ORD-2", 84.5], "kwargs": {}},
        },
        timeout=10.0,
    )
    assert changed.status_code == 409
    assert "ORD-2" in changed.json()["error"]["diff"]


def test_stub_never_runs_live_even_when_policy_is_live(
    fake_llm_server: fake_llm.FakeLLMServer, tmp_path: Path
) -> None:
    db_path = tmp_path / "cassettes.db"
    config = Config(replay=ReplayConfig(tool_miss_policy="live"))
    running = run_daemon(db_path, fake_llm_server.url, config=config, port=0)
    try:
        parent = create_run(running.url)
        child = create_run(
            running.url,
            mode="REPLAY",
            parent_run_id=parent,
            run_seed=int(run_row(db_path, parent)["run_seed"]),
        )
        response = httpx.post(
            f"{running.url}/api/sdk/lookup",
            json={
                "run_id": child,
                "step_type": "TOOL_CALL",
                "name": "send_email",
                "tool_mode": "stub",
                "input": {"args": [], "kwargs": {}},
            },
            timeout=10.0,
        )
        assert response.status_code == 409
        assert run_row(db_path, child)["status"] == "DIVERGED"
    finally:
        running.stop()


def test_pure_tool_live_miss_policy_returns_live(
    fake_llm_server: fake_llm.FakeLLMServer, tmp_path: Path
) -> None:
    db_path = tmp_path / "cassettes.db"
    config = Config(replay=ReplayConfig(tool_miss_policy="live"))
    running = run_daemon(db_path, fake_llm_server.url, config=config, port=0)
    try:
        parent = create_run(running.url)
        child = create_run(
            running.url,
            mode="REPLAY",
            parent_run_id=parent,
            run_seed=int(run_row(db_path, parent)["run_seed"]),
        )
        response = httpx.post(
            f"{running.url}/api/sdk/lookup",
            json={
                "run_id": child,
                "step_type": "TOOL_CALL",
                "name": "lookup_order",
                "tool_mode": "pure",
                "input": {"args": ["ORD-1"], "kwargs": {}},
            },
            timeout=10.0,
        )
        assert response.status_code == 200
        assert response.json()["status"] == "live"
        assert steps_for(db_path, child) == []
    finally:
        running.stop()


def test_recorded_tool_error_is_replayed_as_an_error(daemon: DaemonHandle) -> None:
    parent = create_run(daemon.url)
    tool_input = {"args": [], "kwargs": {}}
    assert post_step(
        daemon.url,
        {
            "run_id": parent,
            "step_type": "TOOL_CALL",
            "name": "flaky",
            "tool_mode": "pure",
            "input": tool_input,
            "error": "RuntimeError: upstream tool failed",
        },
    ).status_code == 200
    child = child_of(daemon, parent)

    response = httpx.post(
        f"{daemon.url}/api/sdk/lookup",
        json={
            "run_id": child,
            "step_type": "TOOL_CALL",
            "name": "flaky",
            "tool_mode": "pure",
            "input": tool_input,
        },
        timeout=10.0,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "replayed_error"
    assert "upstream tool failed" in response.json()["error"]
    child_step = steps_for(daemon.db_path, child)[0]
    assert child_step["source"] == "REPLAYED"
    assert "upstream tool failed" in child_step["error"]


def test_checkpoint_replay_returns_the_recorded_state(daemon: DaemonHandle) -> None:
    parent = create_run(daemon.url)
    state = {"order": {"status": "refunded"}, "amount": 84.5}
    assert post_step(
        daemon.url,
        {
            "run_id": parent,
            "step_type": "CHECKPOINT",
            "name": "after_refund",
            "input": {},
            "output": state,
        },
    ).status_code == 200
    child = child_of(daemon, parent)

    response = httpx.post(
        f"{daemon.url}/api/sdk/lookup",
        json={"run_id": child, "step_type": "CHECKPOINT", "name": "after_refund", "input": {}},
        timeout=10.0,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "replayed"
    assert response.json()["output"] == state
    child_step = steps_for(daemon.db_path, child)[0]
    assert child_step["step_type"] == "CHECKPOINT"
    assert child_step["source"] == "REPLAYED"
    assert child_step["post_state_hash"] == child_step["output_hash"]


def test_run_info_reports_seed_and_root_clock(daemon: DaemonHandle) -> None:
    parent = create_run(daemon.url)
    parent_row = run_row(daemon.db_path, parent)
    child = child_of(daemon, parent)

    parent_info = httpx.get(f"{daemon.url}/api/sdk/run/{parent}", timeout=10.0).json()
    child_info = httpx.get(f"{daemon.url}/api/sdk/run/{child}", timeout=10.0).json()
    assert parent_info["run_seed"] == parent_row["run_seed"]
    assert parent_info["mode"] == "RECORD"
    assert parent_info["freeze_clock"] is True
    assert child_info["mode"] == "REPLAY"
    assert child_info["parent_run_id"] == parent
    assert child_info["run_seed"] == parent_row["run_seed"]
    assert child_info["clock_start"] == parent_info["clock_start"] == parent_row["created_at"]
