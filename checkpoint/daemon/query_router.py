"""Read-only query API and the live event WebSocket."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from checkpoint.daemon.events import EventBus
from checkpoint.daemon.payloads import (
    experiment_payload,
    run_payload,
    step_payload,
)
from checkpoint.daemon.recorder import Recorder
from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import init_db

router = APIRouter()


def _error(message: str, status: int = 404) -> JSONResponse:
    return JSONResponse({"error": {"message": message}}, status_code=status)


def _limit(limit: int) -> int:
    return max(1, min(limit, 500))


def _blob(blobs: BlobStore, blob_hash: Any) -> Any:
    if blob_hash is None:
        return None
    try:
        return blobs.get_json(str(blob_hash))
    except KeyError:
        return None


@router.get("/api/runs")
def list_runs(
    request: Request,
    limit: int = 100,
    parent_run_id: str | None = None,
    experiment_id: str | None = None,
) -> JSONResponse:
    conn = init_db(request.app.state.db_path)
    try:
        recorder = Recorder(conn, BlobStore(conn))
        runs = recorder.list_runs(_limit(limit), parent_run_id, experiment_id)
        return JSONResponse({"runs": [run_payload(run) for run in runs]})
    finally:
        conn.close()


@router.get("/api/runs/{run_id}")
def get_run(run_id: str, request: Request) -> JSONResponse:
    conn = init_db(request.app.state.db_path)
    try:
        recorder = Recorder(conn, BlobStore(conn))
        run = recorder.get_run(run_id)
        if run is None:
            return _error(f"unknown run: {run_id}")
        lineage: list[dict[str, Any]] = []
        seen = {run_id}
        current = run
        while current is not None and current["parent_run_id"] is not None:
            parent_id = str(current["parent_run_id"])
            if parent_id in seen:
                break
            seen.add(parent_id)
            parent = recorder.get_run(parent_id)
            if parent is None:
                break
            lineage.append(run_payload(parent))
            current = parent
        lineage.reverse()
        return JSONResponse(
            {
                "run": run_payload(run),
                "lineage": lineage,
                "children": [
                    run_payload(child) for child in recorder.children_of(run_id)
                ],
                "divergence_diff": recorder.divergence_diff(run_id),
            }
        )
    finally:
        conn.close()


@router.get("/api/runs/{run_id}/steps")
def get_run_steps(run_id: str, request: Request) -> JSONResponse:
    conn = init_db(request.app.state.db_path)
    try:
        recorder = Recorder(conn, BlobStore(conn))
        if recorder.get_run(run_id) is None:
            return _error(f"unknown run: {run_id}")
        steps = [step_payload(step) for step in recorder.steps_for(run_id)]
        return JSONResponse({"run_id": run_id, "steps": steps})
    finally:
        conn.close()


@router.get("/api/steps/{step_id}")
def get_step(step_id: str, request: Request) -> JSONResponse:
    conn = init_db(request.app.state.db_path)
    try:
        recorder = Recorder(conn, BlobStore(conn))
        step = recorder.get_step(step_id)
        if step is None:
            return _error(f"unknown step: {step_id}")
        blobs = BlobStore(conn)
        payload = step_payload(step)
        payload["input"] = _blob(blobs, step["input_hash"])
        payload["output"] = _blob(blobs, step["output_hash"])
        payload["post_state"] = _blob(blobs, step["post_state_hash"])
        return JSONResponse(payload)
    finally:
        conn.close()


@router.get("/api/experiments")
def list_experiments(request: Request, limit: int = 100) -> JSONResponse:
    conn = init_db(request.app.state.db_path)
    try:
        recorder = Recorder(conn, BlobStore(conn))
        experiments = recorder.list_experiments(_limit(limit))
        return JSONResponse(
            {
                "experiments": [
                    experiment_payload(experiment) for experiment in experiments
                ]
            }
        )
    finally:
        conn.close()


@router.websocket("/ws")
async def event_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    bus: EventBus = websocket.app.state.events
    bus.bind()
    queue = bus.subscribe()
    receive_task = asyncio.create_task(websocket.receive())
    try:
        while True:
            queue_task = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait(
                {receive_task, queue_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if receive_task in done:
                queue_task.cancel()
                try:
                    await receive_task
                except (WebSocketDisconnect, RuntimeError):
                    pass
                break
            event = queue_task.result()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(queue)
        if not receive_task.done():
            receive_task.cancel()
