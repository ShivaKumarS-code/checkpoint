"""Query API tests using httpx.AsyncClient against the ASGI app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from checkpoint.daemon.app import create_app
from checkpoint.daemon.recorder import Recorder
from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import init_db


@pytest.fixture()
def seeded_app(tmp_path: Path) -> Any:
    app = create_app(tmp_path / "cassettes.db", "http://127.0.0.1:1")
    conn = init_db(app.state.db_path)
    recorder = Recorder(conn, BlobStore(conn))
    parent = recorder.create_run("parent", "RECORD")
    parent_id = str(parent["run_id"])
    recorder.record_llm_step(
        parent_id,
        "plan",
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        {"choices": [{"message": {"content": "ok"}}]},
        latency_ms=3.0,
    )
    recorder.record_tool_step(
        parent_id, "lookup", {"args": [1], "kwargs": {}}, {"result": 2}
    )
    recorder.record_checkpoint(parent_id, "after", {"refund": None})
    recorder.finish_run(parent_id, "COMPLETED")
    child = recorder.create_run(
        "child", "FORK", parent_run_id=parent_id, forked_at_step=1, variant_name="v1"
    )
    child_id = str(child["run_id"])
    recorder.finish_run(child_id, "COMPLETED")
    experiment_id = recorder.create_experiment(
        "exp", parent_id, 1, [{"name": "v1"}], [{"type": "no_errors"}]
    )
    conn.close()
    step_id = ""
    conn = init_db(app.state.db_path)
    try:
        row = conn.execute(
            "SELECT step_id FROM steps WHERE run_id = ? ORDER BY step_index LIMIT 1",
            (parent_id,),
        ).fetchone()
        step_id = str(row["step_id"])
    finally:
        conn.close()
    yield app, parent_id, child_id, experiment_id, step_id
    app.state.recorder._conn.close()


@pytest.fixture()
async def client(seeded_app: Any) -> Any:
    app: FastAPI = seeded_app[0]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


async def test_root_health_docs_and_openapi(client: httpx.AsyncClient) -> None:
    root = await client.get("/")
    assert root.status_code == 200
    assert root.json()["name"] == "checkpoint"
    assert (await client.get("/api/health")).json() == {"status": "ok"}
    docs = await client.get("/docs")
    assert docs.status_code == 200
    assert "swagger" in docs.text.lower()
    spec = (await client.get("/openapi.json")).json()
    for path in (
        "/api/runs",
        "/api/runs/{run_id}",
        "/api/runs/{run_id}/steps",
        "/api/steps/{step_id}",
        "/api/experiments",
        "/api/health",
    ):
        assert path in spec["paths"]
    assert "/ws" not in spec["paths"]


async def test_list_runs_filters_and_counts(
    client: httpx.AsyncClient, seeded_app: Any
) -> None:
    _, parent_id, child_id, experiment_id, _ = seeded_app
    response = await client.get("/api/runs")
    assert response.status_code == 200
    runs = {run["run_id"]: run for run in response.json()["runs"]}
    assert set(runs) == {parent_id, child_id}
    assert runs[parent_id]["step_count"] == 3
    assert runs[child_id]["step_count"] == 0
    assert runs[child_id]["parent_run_id"] == parent_id
    assert runs[child_id]["variant_name"] == "v1"

    children = (await client.get("/api/runs", params={"parent_run_id": parent_id})).json()
    assert [run["run_id"] for run in children["runs"]] == [child_id]
    experiment = (
        await client.get("/api/runs", params={"experiment_id": experiment_id})
    ).json()
    assert experiment["runs"] == []
    limited = (await client.get("/api/runs", params={"limit": 1})).json()
    assert len(limited["runs"]) == 1


async def test_run_detail_lineage_children_and_404(
    client: httpx.AsyncClient, seeded_app: Any
) -> None:
    _, parent_id, child_id, _, _ = seeded_app
    detail = (await client.get(f"/api/runs/{child_id}")).json()
    assert detail["run"]["run_id"] == child_id
    assert [run["run_id"] for run in detail["lineage"]] == [parent_id]
    assert detail["children"] == []
    assert detail["divergence_diff"] is None

    parent = (await client.get(f"/api/runs/{parent_id}")).json()
    assert [child["run_id"] for child in parent["children"]] == [child_id]
    assert parent["lineage"] == []

    assert (await client.get("/api/runs/nope")).status_code == 404
    error = (await client.get("/api/runs/nope")).json()["error"]["message"]
    assert "unknown run" in error


async def test_run_steps_and_step_payload(
    client: httpx.AsyncClient, seeded_app: Any
) -> None:
    _, parent_id, _, _, step_id = seeded_app
    steps = (await client.get(f"/api/runs/{parent_id}/steps")).json()["steps"]
    assert [step["name"] for step in steps] == ["plan", "lookup", "after"]
    assert [step["step_index"] for step in steps] == [0, 1, 2]
    assert [step["step_type"] for step in steps] == [
        "LLM_CALL",
        "TOOL_CALL",
        "CHECKPOINT",
    ]
    assert "input" not in steps[0]
    assert steps[0]["step_id"] == step_id

    detail = (await client.get(f"/api/steps/{step_id}")).json()
    assert detail["name"] == "plan"
    assert detail["input"]["messages"][0]["content"] == "hi"
    assert detail["output"]["choices"][0]["message"]["content"] == "ok"
    assert detail["post_state"] is None

    checkpoint = (await client.get(f"/api/runs/{parent_id}/steps")).json()["steps"][2]
    checkpoint_detail = (await client.get(f"/api/steps/{checkpoint['step_id']}")).json()
    assert checkpoint_detail["post_state"] == {"refund": None}

    assert (await client.get("/api/steps/nope")).status_code == 404
    assert (await client.get("/api/runs/nope/steps")).status_code == 404


async def test_experiments_list_and_unknown_detail(
    client: httpx.AsyncClient, seeded_app: Any
) -> None:
    _, parent_id, _, experiment_id, _ = seeded_app
    experiments = (await client.get("/api/experiments")).json()["experiments"]
    assert [experiment["experiment_id"] for experiment in experiments] == [experiment_id]
    assert experiments[0]["parent_run_id"] == parent_id
    assert experiments[0]["fork_step_index"] == 1
    assert experiments[0]["run_count"] == 0
    assert (await client.get("/api/experiments/nope")).status_code == 404
