"""SDK replay tests: pure/stub/live tools, checkpoints and entropy."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

import checkpoint
from checkpoint.config import Config, ReplayConfig
from checkpoint.daemon.app import run_daemon
from checkpoint.sdk import client as sdk_client
from examples import fake_llm

from .conftest import DaemonHandle
from .test_proxy_record import create_run
from .test_replay import run_row, steps_for


def bind(
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    mode: str,
    run_id: str | None = None,
) -> None:
    monkeypatch.setenv("CHECKPOINT_DAEMON_URL", url)
    monkeypatch.setenv("CHECKPOINT_MODE", mode)
    if run_id is None:
        monkeypatch.delenv("CHECKPOINT_RUN_ID", raising=False)
    else:
        monkeypatch.setenv("CHECKPOINT_RUN_ID", run_id)


def child_of(daemon: DaemonHandle, parent: str) -> str:
    return create_run(
        daemon.url,
        mode="REPLAY",
        parent_run_id=parent,
        run_seed=int(run_row(daemon.db_path, parent)["run_seed"]),
    )


def test_replayed_pure_tool_and_entropy_match_the_recording(
    daemon: DaemonHandle, monkeypatch: pytest.MonkeyPatch
) -> None:
    executions = []
    bind(monkeypatch, daemon.url, "record")
    checkpoint.init(run_name="sdk-replay")
    state = sdk_client.run_state()
    assert state is not None
    parent = state.run_id

    @checkpoint.tool(mode="pure")
    def first() -> int:
        executions.append(1)
        return 1

    assert first() == 1
    stamp = {"now": datetime.now(UTC).isoformat(), "id": uuid.uuid4().hex}
    checkpoint.checkpoint("stamp", stamp)
    assert executions == [1]

    sdk_client.reset()
    child = child_of(daemon, parent)
    bind(monkeypatch, daemon.url, "replay", child)
    checkpoint.init()

    assert first() == 1
    assert executions == [1], "pure tool must not execute during replay"
    replay_stamp = {"now": datetime.now(UTC).isoformat(), "id": uuid.uuid4().hex}
    assert replay_stamp == stamp
    assert checkpoint.checkpoint("stamp", {"now": "ignored"}) == stamp

    child_steps = steps_for(daemon.db_path, child)
    assert [(row["step_type"], row["name"]) for row in child_steps] == [
        ("TOOL_CALL", "first"),
        ("CHECKPOINT", "stamp"),
    ]
    assert all(row["source"] == "REPLAYED" for row in child_steps)


def test_live_tool_executes_and_records_live_during_replay(
    daemon: DaemonHandle, monkeypatch: pytest.MonkeyPatch
) -> None:
    executions: list[int] = []
    bind(monkeypatch, daemon.url, "record")
    checkpoint.init(run_name="sdk-replay")

    @checkpoint.tool(mode="live")
    def fetch_value() -> int:
        executions.append(1)
        return 1

    assert fetch_value() == 1
    state = sdk_client.run_state()
    assert state is not None
    parent = state.run_id
    sdk_client.reset()

    child = child_of(daemon, parent)
    bind(monkeypatch, daemon.url, "replay", child)
    checkpoint.init()

    executions.clear()
    assert fetch_value() == 1
    assert len(executions) == 1, "live tool must execute during replay"
    step = steps_for(daemon.db_path, child)[0]
    assert step["source"] == "LIVE"
    assert step["tool_mode"] == "live"


def test_stub_miss_raises_and_never_executes(
    daemon: DaemonHandle, monkeypatch: pytest.MonkeyPatch
) -> None:
    executions: list[int] = []
    parent = create_run(daemon.url)
    child = child_of(daemon, parent)
    bind(monkeypatch, daemon.url, "replay", child)
    checkpoint.init()

    @checkpoint.tool(mode="stub")
    def send_email() -> None:
        executions.append(1)

    with pytest.raises(checkpoint.DivergenceError):
        send_email()
    assert executions == []
    assert run_row(daemon.db_path, child)["status"] == "DIVERGED"


def test_pure_miss_under_live_policy_executes_and_records_live(
    fake_llm_server: fake_llm.FakeLLMServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "cassettes.db"
    config = Config(replay=ReplayConfig(tool_miss_policy="live"))
    running = run_daemon(db_path, fake_llm_server.url, config=config, port=0)
    executions: list[int] = []
    try:
        parent = create_run(running.url)
        parent_row = run_row(db_path, parent)
        child = create_run(
            running.url,
            mode="REPLAY",
            parent_run_id=parent,
            run_seed=int(parent_row["run_seed"]),
        )
        bind(monkeypatch, running.url, "replay", child)
        checkpoint.init()

        @checkpoint.tool(mode="pure")
        def double(x: int) -> int:
            executions.append(1)
            return x * 2

        assert double(3) == 6
        assert executions == [1]
        step = steps_for(db_path, child)[0]
        assert step["source"] == "LIVE"
    finally:
        running.stop()
