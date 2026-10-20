"""Tests for the recording proxy."""

from __future__ import annotations

import socket
import sqlite3
from pathlib import Path
from typing import Any

import httpx

from checkpoint.daemon.app import run_daemon
from checkpoint.matching.normalize import match_key
from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import connect
from examples import fake_llm

from .conftest import DaemonHandle

TEST_API_KEY = "sk-test-0123456789abcdef0123"


def create_run(
    url: str,
    name: str = "proxy-test",
    mode: str = "RECORD",
    parent_run_id: str | None = None,
    run_seed: int | None = None,
) -> str:
    response = httpx.post(
        f"{url}/api/sdk/run/start",
        json={
            "name": name,
            "mode": mode,
            "parent_run_id": parent_run_id,
            "run_seed": run_seed,
        },
        timeout=10.0,
    )
    response.raise_for_status()
    return str(response.json()["run_id"])


def chat_body(content: str = "hello") -> dict[str, Any]:
    return {
        "model": "demo-model",
        "temperature": 0.0,
        "messages": [{"role": "user", "content": content}],
    }


def post_chat(
    url: str,
    body: dict[str, Any],
    run_id: str | None = None,
    step_name: str | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    call_headers = dict(headers or {})
    if run_id:
        call_headers["x-checkpoint-run-id"] = run_id
    if step_name:
        call_headers["x-checkpoint-step-name"] = step_name
    return httpx.post(
        f"{url}/v1/chat/completions", json=body, headers=call_headers, timeout=30.0
    )


def stream_chat(
    url: str,
    body: dict[str, Any],
    run_id: str | None = None,
    step_name: str | None = None,
) -> tuple[int, httpx.Headers, list[bytes]]:
    call_headers: dict[str, str] = {}
    if run_id:
        call_headers["x-checkpoint-run-id"] = run_id
    if step_name:
        call_headers["x-checkpoint-step-name"] = step_name
    with httpx.stream(
        "POST",
        f"{url}/v1/chat/completions",
        json=body,
        headers=call_headers,
        timeout=30.0,
    ) as response:
        chunks = [chunk for chunk in response.iter_raw() if chunk]
        return response.status_code, response.headers, chunks


def post_messages(
    url: str,
    body: dict[str, Any],
    run_id: str | None = None,
    step_name: str | None = None,
) -> httpx.Response:
    headers: dict[str, str] = {}
    if run_id:
        headers["x-checkpoint-run-id"] = run_id
    if step_name:
        headers["x-checkpoint-step-name"] = step_name
    return httpx.post(f"{url}/v1/messages", json=body, headers=headers, timeout=30.0)


def stream_messages(
    url: str,
    body: dict[str, Any],
    run_id: str | None = None,
    step_name: str | None = None,
) -> tuple[int, list[bytes]]:
    headers: dict[str, str] = {}
    if run_id:
        headers["x-checkpoint-run-id"] = run_id
    if step_name:
        headers["x-checkpoint-step-name"] = step_name
    with httpx.stream(
        "POST", f"{url}/v1/messages", json=body, headers=headers, timeout=30.0
    ) as response:
        return response.status_code, [chunk for chunk in response.iter_raw() if chunk]


def fetch_steps(db_path: Path) -> list[sqlite3.Row]:
    conn = connect(db_path)
    try:
        return conn.execute("SELECT * FROM steps ORDER BY step_index").fetchall()
    finally:
        conn.close()


def test_proxy_records_an_llm_step_with_match_key(daemon: DaemonHandle) -> None:
    run_id = create_run(daemon.url)
    body = chat_body("please refund ORD-1002")
    response = post_chat(daemon.url, body, run_id=run_id, step_name="plan")

    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"
    steps = fetch_steps(daemon.db_path)
    assert len(steps) == 1
    step = steps[0]
    assert step["step_index"] == 0
    assert step["step_type"] == "LLM_CALL"
    assert step["name"] == "plan"
    assert step["occurrence"] == 0
    assert step["source"] == "RECORDED"
    assert step["match_key"] == match_key(body)
    assert step["latency_ms"] >= 0.0
    assert step["error"] is None

    conn = connect(daemon.db_path)
    try:
        blobs = BlobStore(conn)
        assert blobs.get_json(step["input_hash"]) == body
        output = blobs.get_json(step["output_hash"])
        assert output["choices"][0]["message"]["content"].startswith("This is a deterministic")
    finally:
        conn.close()


def test_default_step_name_and_occurrence_counting(daemon: DaemonHandle) -> None:
    run_id = create_run(daemon.url)
    post_chat(daemon.url, chat_body("one"), run_id=run_id)
    post_chat(daemon.url, chat_body("two"), run_id=run_id)
    steps = fetch_steps(daemon.db_path)
    assert [(step["name"], step["occurrence"]) for step in steps] == [
        ("llm_call", 0),
        ("llm_call", 1),
    ]


def test_missing_run_header_is_rejected(daemon: DaemonHandle) -> None:
    response = post_chat(daemon.url, chat_body())
    assert response.status_code == 400
    assert "checkpoint.init" in response.json()["error"]["message"]


def test_unknown_run_is_rejected(daemon: DaemonHandle) -> None:
    response = post_chat(daemon.url, chat_body(), run_id="does-not-exist")
    assert response.status_code == 400


def test_streaming_records_and_replays_identical_chunks(
    daemon: DaemonHandle, fake_llm_server: fake_llm.FakeLLMServer
) -> None:
    run_id = create_run(daemon.url)
    body = {**chat_body("stream this response"), "stream": True}
    status, headers, recorded = stream_chat(
        daemon.url, body, run_id=run_id, step_name="plan"
    )
    assert status == 200
    assert recorded
    assert "text/event-stream" in headers["content-type"]

    steps = fetch_steps(daemon.db_path)
    assert len(steps) == 1
    assert steps[0]["step_type"] == "LLM_CALL"
    assert steps[0]["source"] == "RECORDED"
    conn = connect(daemon.db_path)
    try:
        output = BlobStore(conn).get_json(steps[0]["output_hash"])
        assert output["__checkpoint_stream__"] is True
        assert output["chunks"]
    finally:
        conn.close()

    fake_llm.reset()
    replay_id = create_run(
        daemon.url,
        name="replay-stream",
        mode="REPLAY",
        parent_run_id=run_id,
        run_seed=1,
    )
    replay_status, replay_headers, replayed = stream_chat(
        daemon.url, body, run_id=replay_id, step_name="plan"
    )
    assert replay_status == 200
    assert replayed == recorded
    assert replay_headers["x-checkpoint-frozen-timestamp"] == headers[
        "x-checkpoint-frozen-timestamp"
    ]
    assert fake_llm.call_count() == 0

    replay_steps = [step for step in fetch_steps(daemon.db_path) if step["run_id"] == replay_id]
    assert [step["source"] for step in replay_steps] == ["REPLAYED"]
    assert replay_steps[0]["output_hash"] == steps[0]["output_hash"]


def test_anthropic_messages_record_and_replay(daemon: DaemonHandle) -> None:
    run_id = create_run(daemon.url)
    body = {
        "model": "claude-demo",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello anthropic"}],
    }
    recorded = post_messages(daemon.url, body, run_id=run_id, step_name="plan")
    assert recorded.status_code == 200
    payload = recorded.json()
    assert payload["type"] == "message"
    assert payload["role"] == "assistant"
    assert payload["content"][0]["type"] == "text"

    replay_id = create_run(
        daemon.url,
        name="replay-anthropic",
        mode="REPLAY",
        parent_run_id=run_id,
        run_seed=1,
    )
    replayed = post_messages(daemon.url, body, run_id=replay_id, step_name="plan")
    assert replayed.status_code == 200
    assert replayed.json() == payload
    steps = [step for step in fetch_steps(daemon.db_path) if step["run_id"] == replay_id]
    assert steps[0]["source"] == "REPLAYED"


def test_anthropic_messages_stream_replays_same_chunks(daemon: DaemonHandle) -> None:
    run_id = create_run(daemon.url)
    body = {
        "model": "claude-demo",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "stream anthropic"}],
    }
    status, recorded = stream_messages(
        daemon.url, body, run_id=run_id, step_name="plan"
    )
    assert status == 200
    assert recorded
    assert any(b"event: message_start" in chunk for chunk in recorded)

    replay_id = create_run(
        daemon.url,
        name="replay-anthropic-stream",
        mode="REPLAY",
        parent_run_id=run_id,
        run_seed=1,
    )
    replay_status, replayed = stream_messages(
        daemon.url, body, run_id=replay_id, step_name="plan"
    )
    assert replay_status == 200
    assert replayed == recorded


def test_run_totals_accumulate_usage(daemon: DaemonHandle) -> None:
    run_id = create_run(daemon.url)
    post_chat(daemon.url, chat_body("count tokens"), run_id=run_id, step_name="plan")
    conn = connect(daemon.db_path)
    try:
        run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        assert run["total_tokens"] > 0
        assert run["total_latency_ms"] >= 0.0
    finally:
        conn.close()


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def test_upstream_failure_is_a_502_and_is_recorded(tmp_path: Path) -> None:
    dead_upstream = f"http://127.0.0.1:{free_port()}"
    db_path = tmp_path / "cassettes.db"
    running = run_daemon(db_path, dead_upstream, port=0)
    try:
        run_id = create_run(running.url)
        response = post_chat(running.url, chat_body(), run_id=run_id)
        assert response.status_code == 502
    finally:
        running.stop()
    steps = fetch_steps(db_path)
    assert len(steps) == 1
    assert steps[0]["output_hash"] is None
    assert "upstream request failed" in steps[0]["error"]


def test_api_key_never_reaches_the_database(daemon: DaemonHandle) -> None:
    run_id = create_run(daemon.url)
    body = chat_body(f"my key is {TEST_API_KEY} please help")
    response = post_chat(
        daemon.url,
        body,
        run_id=run_id,
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
    )
    assert response.status_code == 200

    raw = daemon.db_path.read_bytes()
    assert TEST_API_KEY.encode() not in raw

    conn = connect(daemon.db_path)
    try:
        blobs = BlobStore(conn)
        for row in conn.execute("SELECT * FROM blobs"):
            assert TEST_API_KEY.encode() not in blobs.get(row["blob_hash"])
        for table in ("runs", "steps", "experiments", "verdicts"):
            for row in conn.execute(f"SELECT * FROM {table}"):
                assert all(TEST_API_KEY not in str(value) for value in tuple(row))
        step = conn.execute("SELECT * FROM steps LIMIT 1").fetchone()
        assert "<redacted>" in str(blobs.get_json(step["input_hash"]))
    finally:
        conn.close()


def test_replay_run_without_parent_diverges(daemon: DaemonHandle) -> None:
    run_id = create_run(daemon.url, name="replay-test", mode="REPLAY")
    response = post_chat(daemon.url, chat_body(), run_id=run_id)
    assert response.status_code == 409
    assert response.json()["error"]["type"] == "checkpoint_divergence"

    conn = connect(daemon.db_path)
    try:
        run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        assert run["status"] == "DIVERGED"
        assert run["divergence_step"] is None
        assert run["divergence_diff_hash"] is not None
    finally:
        conn.close()


def test_health_endpoint(daemon: DaemonHandle) -> None:
    response = httpx.get(f"{daemon.url}/api/health", timeout=10.0)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
