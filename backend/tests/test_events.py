"""WebSocket event tests against a live daemon."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from websockets.sync.client import connect

from checkpoint.daemon.app import RunningDaemon, create_app


@pytest.fixture()
def live_daemon(tmp_path: Path) -> Iterator[tuple[FastAPI, RunningDaemon]]:
    app = create_app(tmp_path / "cassettes.db", "http://127.0.0.1:1")
    daemon = RunningDaemon(app, port=0).start()
    yield app, daemon
    daemon.stop()


def _payloads(ws: Any, count: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for _ in range(count):
        raw = ws.recv(timeout=10)
        events.append(json.loads(raw))
    return events


def test_websocket_events_in_order_during_a_recorded_run(
    live_daemon: tuple[FastAPI, RunningDaemon],
) -> None:
    app, daemon = live_daemon
    with connect(f"ws://127.0.0.1:{daemon.url.rsplit(':', 1)[1]}/ws") as ws:
        created = httpx.post(
            f"{daemon.url}/api/sdk/run/start",
            json={"name": "ws-run", "mode": "RECORD"},
            timeout=10.0,
        )
        assert created.status_code == 200
        run_id = created.json()["run_id"]
        for index in range(2):
            step = httpx.post(
                f"{daemon.url}/api/sdk/step",
                json={
                    "run_id": run_id,
                    "step_type": "CHECKPOINT",
                    "name": f"checkpoint_{index}",
                    "output": {"index": index},
                },
                timeout=10.0,
            )
            assert step.status_code == 200
        app.state.recorder.finish_run(run_id, "COMPLETED")
        events = _payloads(ws, 4)
    for _ in range(100):
        if app.state.events.subscriber_count() == 0:
            break
        time.sleep(0.01)
    assert app.state.events.subscriber_count() == 0

    assert [event["type"] for event in events] == [
        "run_started",
        "step_added",
        "step_added",
        "run_finished",
    ]
    assert events[0]["run"]["run_id"] == run_id
    assert [event["step"]["step_index"] for event in events[1:3]] == [0, 1]
    assert [event["step"]["name"] for event in events[1:3]] == [
        "checkpoint_0",
        "checkpoint_1",
    ]
    assert events[3]["run"]["status"] == "COMPLETED"


def test_websocket_receives_experiment_progress(
    live_daemon: tuple[FastAPI, RunningDaemon], tmp_path: Path
) -> None:
    app, daemon = live_daemon
    parent = app.state.recorder.create_run("parent", "RECORD")
    parent_id = str(parent["run_id"])
    script = tmp_path / "variant.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    spec = {
        "fork_at": 0,
        "command": f'"{sys.executable}" "{script}"',
        "variants": [{"name": "only"}],
        "checks": [],
        "max_parallel": 1,
    }
    with connect(f"ws://127.0.0.1:{daemon.url.rsplit(':', 1)[1]}/ws") as ws:
        response = httpx.post(
            f"{daemon.url}/api/experiments",
            json={
                "parent_run_id": parent_id,
                "spec": spec,
                "base_dir": str(tmp_path),
            },
            timeout=60.0,
        )
        assert response.status_code == 200, response.text
        events = _payloads(ws, 5)

    assert [event["type"] for event in events] == [
        "experiment_started",
        "run_started",
        "run_finished",
        "experiment_progress",
        "experiment_finished",
    ]
    experiment_id = response.json()["experiment_id"]
    assert events[0]["experiment"]["experiment_id"] == experiment_id
    assert events[1]["run"]["mode"] == "FORK"
    assert events[1]["run"]["experiment_id"] == experiment_id
    assert events[2]["run"]["status"] == "COMPLETED"
    assert events[3]["variant_name"] == "only"
    assert events[3]["completed"] == 1
    assert events[3]["total"] == 1
    assert events[4]["experiment_id"] == experiment_id
    assert events[4]["passed"] == 1
    assert events[4]["failed"] == 0
