"""SDK RPC endpoints: run registration and tool/checkpoint step recording."""

from __future__ import annotations

from typing import Any, Literal

import msgspec
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from checkpoint.storage.models import RunMode, ToolMode

router = APIRouter()


class StartRunRequest(msgspec.Struct):
    name: str = "run"
    mode: RunMode = "RECORD"


class StepRequest(msgspec.Struct):
    run_id: str
    name: str
    step_type: Literal["TOOL_CALL", "CHECKPOINT"] = "TOOL_CALL"
    tool_mode: ToolMode | None = None
    input: Any = None
    output: Any = None
    error: str | None = None
    latency_ms: float = 0.0


def _bad_request(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": {"message": message}}, status_code=status)


@router.post("/api/sdk/run/start")
async def start_run(request: Request) -> JSONResponse:
    try:
        payload = msgspec.json.decode(await request.body(), type=StartRunRequest)
    except msgspec.ValidationError as exc:
        return _bad_request(str(exc))
    created = request.app.state.recorder.create_run(payload.name, payload.mode)
    return JSONResponse(created)


@router.post("/api/sdk/step")
async def record_step(request: Request) -> JSONResponse:
    try:
        payload = msgspec.json.decode(await request.body(), type=StepRequest)
    except msgspec.ValidationError as exc:
        return _bad_request(str(exc))
    recorder = request.app.state.recorder
    run = recorder.get_run(payload.run_id)
    if run is None:
        return _bad_request(f"unknown run: {payload.run_id}", status=404)
    if run["mode"] != "RECORD":
        return JSONResponse(
            {"error": {"message": f"run mode {run['mode']} does not record steps yet"}},
            status_code=501,
        )
    if payload.step_type == "CHECKPOINT":
        step_index = recorder.record_checkpoint(payload.run_id, payload.name, payload.output)
    else:
        step_index = recorder.record_tool_step(
            payload.run_id,
            payload.name,
            payload.input,
            payload.output,
            tool_mode=payload.tool_mode,
            error=payload.error,
            latency_ms=payload.latency_ms,
        )
    return JSONResponse({"step_index": step_index})
