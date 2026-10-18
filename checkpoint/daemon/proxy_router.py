"""The recording proxy for OpenAI-compatible chat completions."""

from __future__ import annotations

import time
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from checkpoint.daemon.recorder import DEFAULT_LLM_STEP_NAME

router = APIRouter()

_HOP_BY_HOP = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}


def _forward_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        lower = key.lower()
        if lower in _HOP_BY_HOP or lower.startswith("x-checkpoint-"):
            continue
        headers[key] = value
    return headers


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    recorder = request.app.state.recorder
    run_id = request.headers.get("x-checkpoint-run-id")
    if not run_id:
        return JSONResponse(
            {"error": {"message": "missing x-checkpoint-run-id header; call checkpoint.init()"}},
            status_code=400,
        )
    run = recorder.get_run(run_id)
    if run is None:
        return JSONResponse(
            {"error": {"message": f"unknown run: {run_id}"}}, status_code=400
        )
    if run["mode"] != "RECORD":
        return JSONResponse(
            {"error": {"message": f"run mode {run['mode']} is not supported yet"}},
            status_code=501,
        )
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"error": {"message": "invalid JSON body"}}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": {"message": "body must be a JSON object"}}, status_code=400)
    if body.get("stream"):
        return JSONResponse(
            {"error": {"message": "streaming is not supported until Phase 6"}}, status_code=400
        )

    step_name = request.headers.get("x-checkpoint-step-name") or DEFAULT_LLM_STEP_NAME
    upstream_url = f"{request.app.state.upstream_url}/v1/chat/completions"
    started = time.perf_counter()
    try:
        upstream = await request.app.state.http.post(
            upstream_url, json=body, headers=_forward_headers(request)
        )
    except httpx.HTTPError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        recorder.record_llm_step(
            run_id,
            step_name,
            body,
            None,
            latency_ms,
            error=f"upstream request failed: {type(exc).__name__}",
        )
        return JSONResponse(
            {"error": {"message": "upstream request failed", "type": "checkpoint_upstream_error"}},
            status_code=502,
        )
    latency_ms = (time.perf_counter() - started) * 1000.0

    upstream_json: Any = None
    try:
        upstream_json = upstream.json()
    except ValueError:
        upstream_json = None
    error = None if upstream.is_success else f"upstream status {upstream.status_code}"
    if upstream_json is None and error is None:
        error = "upstream returned a non-JSON response"
    recorder.record_llm_step(run_id, step_name, body, upstream_json, latency_ms, error=error)

    if upstream_json is None:
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/octet-stream"),
        )
    return JSONResponse(upstream_json, status_code=upstream.status_code)
