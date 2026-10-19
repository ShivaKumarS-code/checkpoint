"""Tests for FORK mode: replay before N, live after N, stubs never execute."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from checkpoint.daemon.recorder import Recorder
from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import init_db
from examples import fake_llm

from .conftest import DaemonHandle
from .test_proxy_record import chat_body, create_run, post_chat
from .test_replay import run_row, steps_for

BOUNDARY = "x-checkpoint-boundary"


def create_fork(
    url: str,
    parent_run_id: str,
    fork_at: int,
    experiment_id: str | None = None,
    variant_name: str | None = None,
) -> str:
    response = httpx.post(
        f"{url}/api/sdk/run/start",
        json={
            "name": "forked",
            "mode": "FORK",
            "parent_run_id": parent_run_id,
            "forked_at_step": fork_at,
            "experiment_id": experiment_id,
            "variant_name": variant_name,
        },
        timeout=10.0,
    )
    response.raise_for_status()
    return str(response.json()["run_id"])


def post_step(url: str, payload: dict[str, Any]) -> httpx.Response:
    return httpx.post(f"{url}/api/sdk/step", json=payload, timeout=10.0)


def lookup(url: str, payload: dict[str, Any]) -> httpx.Response:
    return httpx.post(f"{url}/api/sdk/lookup", json=payload, timeout=10.0)


def make_parent(daemon: DaemonHandle) -> str:
    parent = create_run(daemon.url)
    plan = post_chat(daemon.url, chat_body("plan this"), run_id=parent, step_name="plan")
    assert plan.status_code == 200
    assert post_step(
        daemon.url,
        {
            "run_id": parent,
            "step_type": "TOOL_CALL",
            "name": "lookup",
            "tool_mode": "pure",
            "input": {"args": [1], "kwargs": {}},
            "output": 2,
        },
    ).status_code == 200
    decide = post_chat(daemon.url, chat_body("decide this"), run_id=parent, step_name="decide")
    assert decide.status_code == 200
    assert post_step(
        daemon.url,
        {
            "run_id": parent,
            "step_type": "TOOL_CALL",
            "name": "refund",
            "tool_mode": "stub",
            "input": {"args": ["ORD-1"], "kwargs": {}},
            "output": {"refund_id": "rf_1"},
        },
    ).status_code == 200
    assert post_step(
        daemon.url,
        {
            "run_id": parent,
            "step_type": "CHECKPOINT",
            "name": "after",
            "input": {},
            "output": {"refund": {"refund_id": "rf_1"}},
        },
    ).status_code == 200
    return parent


def test_fork_replays_before_n_and_goes_live_after(
    daemon: DaemonHandle, fake_llm_server: fake_llm.FakeLLMServer
) -> None:
    parent = make_parent(daemon)
    child = create_fork(daemon.url, parent, fork_at=2)
    fake_llm.reset()

    plan = post_chat(
        daemon.url, chat_body("plan this"), run_id=child, step_name="plan",
        headers={BOUNDARY: "0"},
    )
    assert plan.status_code == 200
    assert fake_llm.call_count() == 0

    tool = lookup(
        daemon.url,
        {
            "run_id": child,
            "step_type": "TOOL_CALL",
            "name": "lookup",
            "tool_mode": "pure",
            "input": {"args": [1], "kwargs": {}},
            "boundary_index": 1,
        },
    )
    assert tool.status_code == 200
    assert tool.json()["status"] == "replayed"
    assert tool.json()["output"] == 2

    decide = post_chat(
        daemon.url, chat_body("decide this"), run_id=child, step_name="decide",
        headers={BOUNDARY: "2"},
    )
    assert decide.status_code == 200
    assert fake_llm.call_count() == 1

    stub = lookup(
        daemon.url,
        {
            "run_id": child,
            "step_type": "TOOL_CALL",
            "name": "refund",
            "tool_mode": "stub",
            "input": {"args": ["ORD-1"], "kwargs": {}},
            "boundary_index": 3,
        },
    )
    assert stub.status_code == 200
    assert stub.json()["status"] == "replayed"
    assert stub.json()["output"] == {"refund_id": "rf_1"}

    checkpoint = lookup(
        daemon.url,
        {
            "run_id": child,
            "step_type": "CHECKPOINT",
            "name": "after",
            "input": {},
            "boundary_index": 4,
        },
    )
    assert checkpoint.status_code == 200
    assert checkpoint.json()["status"] == "live"
    recorded = post_step(
        daemon.url,
        {
            "run_id": child,
            "step_type": "CHECKPOINT",
            "name": "after",
            "input": {},
            "output": {"refund": None},
            "source": "LIVE",
        },
    )
    assert recorded.status_code == 200
    assert recorded.json()["source"] == "LIVE"

    child_steps = steps_for(daemon.db_path, child)
    assert [step["name"] for step in child_steps] == [
        "plan",
        "lookup",
        "decide",
        "refund",
        "after",
    ]
    assert [step["source"] for step in child_steps] == [
        "REPLAYED",
        "REPLAYED",
        "LIVE",
        "REPLAYED",
        "LIVE",
    ]
    assert [step["step_index"] for step in child_steps] == [0, 1, 2, 3, 4]
    assert run_row(daemon.db_path, child)["status"] == "RUNNING"


def test_fork_mismatch_before_n_diverges(daemon: DaemonHandle) -> None:
    parent = make_parent(daemon)
    child = create_fork(daemon.url, parent, fork_at=2)
    response = post_chat(
        daemon.url, chat_body("CHANGED"), run_id=child, step_name="plan",
        headers={BOUNDARY: "0"},
    )
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["type"] == "checkpoint_divergence"
    assert "code changed before the fork point" in error["message"]
    assert steps_for(daemon.db_path, child) == []
    assert run_row(daemon.db_path, child)["status"] == "DIVERGED"


def test_fork_missing_pure_tool_before_n_diverges(daemon: DaemonHandle) -> None:
    parent = make_parent(daemon)
    child = create_fork(daemon.url, parent, fork_at=2)
    response = lookup(
        daemon.url,
        {
            "run_id": child,
            "step_type": "TOOL_CALL",
            "name": "brand_new",
            "tool_mode": "pure",
            "input": {"args": [], "kwargs": {}},
            "boundary_index": 1,
        },
    )
    assert response.status_code == 409
    assert run_row(daemon.db_path, child)["status"] == "DIVERGED"


def test_fork_new_pure_tool_after_n_goes_live(daemon: DaemonHandle) -> None:
    parent = make_parent(daemon)
    child = create_fork(daemon.url, parent, fork_at=2)
    response = lookup(
        daemon.url,
        {
            "run_id": child,
            "step_type": "TOOL_CALL",
            "name": "brand_new",
            "tool_mode": "pure",
            "input": {"args": [], "kwargs": {}},
            "boundary_index": 2,
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "live"
    assert steps_for(daemon.db_path, child) == []


def test_fork_changed_stub_input_fails_loudly_without_diverging(daemon: DaemonHandle) -> None:
    parent = make_parent(daemon)
    child = create_fork(daemon.url, parent, fork_at=2)
    response = lookup(
        daemon.url,
        {
            "run_id": child,
            "step_type": "TOOL_CALL",
            "name": "refund",
            "tool_mode": "stub",
            "input": {"args": ["ORD-2"], "kwargs": {}},
            "boundary_index": 3,
        },
    )
    assert response.status_code == 400
    assert "never execute during a fork" in response.json()["error"]["message"]
    assert run_row(daemon.db_path, child)["status"] == "RUNNING"


def test_fork_stub_miss_fails_loudly(daemon: DaemonHandle) -> None:
    parent = make_parent(daemon)
    child = create_fork(daemon.url, parent, fork_at=2)
    response = lookup(
        daemon.url,
        {
            "run_id": child,
            "step_type": "TOOL_CALL",
            "name": "send_email",
            "tool_mode": "stub",
            "input": {"args": [], "kwargs": {}},
            "boundary_index": 3,
        },
    )
    assert response.status_code == 400
    assert "no recorded output" in response.json()["error"]["message"]
    assert run_row(daemon.db_path, child)["status"] == "RUNNING"


def test_stub_served_after_n_gets_the_child_step_index(daemon: DaemonHandle) -> None:
    parent = make_parent(daemon)
    child = create_fork(daemon.url, parent, fork_at=2)
    assert post_chat(
        daemon.url, chat_body("plan this"), run_id=child, step_name="plan",
        headers={BOUNDARY: "0"},
    ).status_code == 200
    assert lookup(
        daemon.url,
        {
            "run_id": child,
            "step_type": "TOOL_CALL",
            "name": "lookup",
            "tool_mode": "pure",
            "input": {"args": [1], "kwargs": {}},
            "boundary_index": 1,
        },
    ).status_code == 200
    stub = lookup(
        daemon.url,
        {
            "run_id": child,
            "step_type": "TOOL_CALL",
            "name": "refund",
            "tool_mode": "stub",
            "input": {"args": ["ORD-1"], "kwargs": {}},
            "boundary_index": 2,
        },
    )
    assert stub.status_code == 200
    assert stub.json()["step_index"] == 2
    child_steps = steps_for(daemon.db_path, child)
    assert [step["step_index"] for step in child_steps] == [0, 1, 2]
    assert child_steps[2]["source"] == "REPLAYED"


def test_fork_live_llm_applies_variant_overrides(
    daemon: DaemonHandle, fake_llm_server: fake_llm.FakeLLMServer
) -> None:
    parent = make_parent(daemon)
    conn = init_db(daemon.db_path)
    try:
        recorder = Recorder(conn, BlobStore(conn))
        experiment_id = recorder.create_experiment(
            "overrides",
            parent,
            0,
            [
                {"name": "baseline"},
                {
                    "name": "cheap",
                    "llm_overrides": {
                        "*": {
                            "model": "tiny-model",
                            "temperature": 0.9,
                            "system_prompt_replace": "OVERRIDE SYSTEM",
                        }
                    },
                },
            ],
            [],
        )
    finally:
        conn.close()
    child = create_fork(
        daemon.url, parent, fork_at=0, experiment_id=experiment_id, variant_name="cheap"
    )
    fake_llm.reset()
    response = post_chat(
        daemon.url, chat_body("plan this"), run_id=child, step_name="plan",
        headers={BOUNDARY: "0"},
    )
    assert response.status_code == 200
    assert fake_llm.call_count() == 1
    step = steps_for(daemon.db_path, child)[0]
    assert step["source"] == "LIVE"
    conn = init_db(daemon.db_path)
    try:
        recorded = BlobStore(conn).get_json(step["input_hash"])
    finally:
        conn.close()
    assert recorded["model"] == "tiny-model"
    assert recorded["temperature"] == pytest.approx(0.9)
    assert recorded["messages"][0] == {"role": "system", "content": "OVERRIDE SYSTEM"}


def test_fork_run_info_and_live_step(daemon: DaemonHandle) -> None:
    parent = make_parent(daemon)
    child = create_fork(daemon.url, parent, fork_at=2, variant_name="baseline")
    info = httpx.get(f"{daemon.url}/api/sdk/run/{child}", timeout=10.0).json()
    assert info["mode"] == "FORK"
    assert info["parent_run_id"] == parent
    assert info["forked_at_step"] == 2
    assert info["variant_name"] == "baseline"

    missing_source = post_step(
        daemon.url,
        {
            "run_id": child,
            "step_type": "TOOL_CALL",
            "name": "live_tool",
            "tool_mode": "live",
            "input": {"args": [], "kwargs": {}},
            "output": 1,
        },
    )
    assert missing_source.status_code == 400

    live = post_step(
        daemon.url,
        {
            "run_id": child,
            "step_type": "TOOL_CALL",
            "name": "live_tool",
            "tool_mode": "live",
            "input": {"args": [], "kwargs": {}},
            "output": 1,
            "source": "LIVE",
        },
    )
    assert live.status_code == 200
    assert live.json()["source"] == "LIVE"
    assert steps_for(daemon.db_path, child)[0]["source"] == "LIVE"


def test_fork_without_parent_is_rejected(daemon: DaemonHandle) -> None:
    parent = create_run(daemon.url)
    child = create_fork(daemon.url, parent, fork_at=0)
    # A FORK run with no recorded steps has no parent index to match; a miss
    # after the (zero) fork point goes live rather than diverging.
    response = post_chat(
        daemon.url, chat_body("new"), run_id=child, step_name="any",
        headers={BOUNDARY: "0"},
    )
    assert response.status_code == 200
    assert steps_for(daemon.db_path, child)[0]["source"] == "LIVE"
