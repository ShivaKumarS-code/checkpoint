"""The recording and replaying proxy for OpenAI-compatible chat completions.

RECORD forwards and records. REPLAY serves from the cassette. FORK serves the
parent's steps before ``forked_at_step`` and goes live from there, applying the
variant's ``llm_overrides`` to the live request.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from checkpoint.daemon.experiments import (
    apply_llm_overrides,
    experiment_variant_overrides,
)
from checkpoint.daemon.recorder import DEFAULT_LLM_STEP_NAME, StepInfo
from checkpoint.matching.replay import step_label
from checkpoint.sdk.determinism import real_now_iso

router = APIRouter()

_HOP_BY_HOP = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}

HEADER_RUN_ID = "x-checkpoint-run-id"
HEADER_STEP_NAME = "x-checkpoint-step-name"
HEADER_FROZEN_TIMESTAMP = "x-checkpoint-frozen-timestamp"
HEADER_RNG_SEED = "x-checkpoint-rng-seed"
HEADER_BOUNDARY = "x-checkpoint-boundary"


def _forward_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        lower = key.lower()
        if lower in _HOP_BY_HOP or lower.startswith("x-checkpoint-"):
            continue
        headers[key] = value
    return headers


def _settle_headers(info: StepInfo) -> dict[str, str]:
    return {
        HEADER_FROZEN_TIMESTAMP: info.frozen_timestamp,
        HEADER_RNG_SEED: str(info.rng_seed),
    }


def _diverged(message: str, diff: str, step_index: int | None) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": "checkpoint_divergence",
                "diff": diff,
            },
            "step_index": step_index,
        },
        status_code=409,
    )


def _boundary(request: Request) -> int | None:
    raw = request.headers.get(HEADER_BOUNDARY)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


async def _request_body(request: Request) -> dict[str, Any] | None:
    try:
        body = await request.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


async def _forward_upstream(
    request: Request, body: dict[str, Any]
) -> tuple[httpx.Response | None, str | None]:
    upstream_url = f"{request.app.state.upstream_url}/v1/chat/completions"
    try:
        upstream = await request.app.state.http.post(
            upstream_url, json=body, headers=_forward_headers(request)
        )
    except httpx.HTTPError as exc:
        return None, f"upstream request failed: {type(exc).__name__}"
    return upstream, None


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    recorder = request.app.state.recorder
    run_id = request.headers.get(HEADER_RUN_ID)
    if not run_id:
        return JSONResponse(
            {"error": {"message": "missing x-checkpoint-run-id header; call checkpoint.init()"}},
            status_code=400,
        )
    run = recorder.get_run(run_id)
    if run is None:
        return JSONResponse({"error": {"message": f"unknown run: {run_id}"}}, status_code=400)

    body = await _request_body(request)
    if body is None:
        return JSONResponse({"error": {"message": "invalid JSON body"}}, status_code=400)
    if body.get("stream"):
        return JSONResponse(
            {"error": {"message": "streaming is not supported until Phase 6"}}, status_code=400
        )

    mode = str(run["mode"])
    if mode == "REPLAY":
        return await _replay_chat(request, run_id, body)
    if mode == "FORK":
        return await _fork_chat(request, run_id, run, body)
    if mode != "RECORD":
        return JSONResponse(
            {"error": {"message": f"run mode {mode} is not supported yet"}},
            status_code=501,
        )
    return await _record_chat(request, run_id, body)


async def _record_chat(request: Request, run_id: str, body: dict[str, Any]) -> Response:
    recorder = request.app.state.recorder
    step_name = request.headers.get(HEADER_STEP_NAME) or DEFAULT_LLM_STEP_NAME
    frozen_timestamp = request.headers.get(HEADER_FROZEN_TIMESTAMP) or real_now_iso()
    upstream, forward_error, latency_ms = await _call_upstream(request, body)
    if upstream is None:
        return _upstream_error(
            request, run_id, step_name, body, forward_error, latency_ms, "RECORDED"
        )
    return _record_upstream_response(
        recorder, run_id, step_name, body, upstream, latency_ms, frozen_timestamp, "RECORDED"
    )


async def _replay_chat(request: Request, run_id: str, body: dict[str, Any]) -> Response:
    recorder = request.app.state.recorder
    step_name = request.headers.get(HEADER_STEP_NAME) or DEFAULT_LLM_STEP_NAME
    lookup = recorder.lookup_llm(run_id, step_name, body)

    if lookup.found and lookup.matched:
        return _serve_replayed_chat(request, run_id, step_name, body, lookup)

    if request.app.state.config.replay.llm_miss_policy == "live":
        frozen_timestamp = real_now_iso()
        upstream, forward_error, latency_ms = await _call_upstream(request, body)
        if upstream is None:
            return _upstream_error(
                request, run_id, step_name, body, forward_error, latency_ms, "LIVE"
            )
        return _record_upstream_response(
            recorder,
            run_id,
            step_name,
            body,
            upstream,
            latency_ms,
            frozen_timestamp,
            "LIVE",
        )

    label = step_label("LLM_CALL", step_name, lookup.occurrence)
    diff = lookup.diff or f"no recorded {label} in the parent run"
    recorder.mark_diverged(run_id, lookup.step_index, diff)
    return _diverged(f"replay diverged at {label}", diff, lookup.step_index)


async def _fork_chat(
    request: Request, run_id: str, run: Any, body: dict[str, Any]
) -> Response:
    """Serve parent steps before N, then call the upstream with variant overrides."""
    recorder = request.app.state.recorder
    step_name = request.headers.get(HEADER_STEP_NAME) or DEFAULT_LLM_STEP_NAME
    lookup = recorder.lookup_llm(run_id, step_name, body)
    fork_index = int(run["forked_at_step"] or 0)
    boundary = _boundary(request)

    if lookup.found and lookup.step_index is not None and lookup.step_index < fork_index:
        if lookup.matched:
            return _serve_replayed_chat(request, run_id, step_name, body, lookup)
        label = step_label("LLM_CALL", step_name, lookup.occurrence)
        diff = lookup.diff or f"no recorded {label} in the parent run"
        recorder.mark_diverged(run_id, lookup.step_index, diff)
        return _diverged(
            f"code changed before the fork point: {label}", diff, lookup.step_index
        )

    if not lookup.found and boundary is not None and boundary < fork_index:
        label = step_label("LLM_CALL", step_name, lookup.occurrence)
        diff = lookup.diff or f"no recorded {label} in the parent run"
        recorder.mark_diverged(run_id, lookup.step_index, diff)
        return _diverged(
            f"missing step before the fork point: {label}", diff, lookup.step_index
        )

    overrides = experiment_variant_overrides(
        recorder, run["experiment_id"], run["variant_name"]
    )
    effective = apply_llm_overrides(body, step_name, overrides)
    frozen_timestamp = request.headers.get(HEADER_FROZEN_TIMESTAMP) or real_now_iso()
    upstream, forward_error, latency_ms = await _call_upstream(request, effective)
    if upstream is None:
        return _upstream_error(
            request, run_id, step_name, effective, forward_error, latency_ms, "LIVE"
        )
    return _record_upstream_response(
        recorder,
        run_id,
        step_name,
        effective,
        upstream,
        latency_ms,
        frozen_timestamp,
        "LIVE",
    )


def _serve_replayed_chat(
    request: Request, run_id: str, step_name: str, body: dict[str, Any], lookup: Any
) -> Response:
    recorder = request.app.state.recorder
    output: Any = None
    if lookup.step is not None and lookup.step["output_hash"] is not None:
        output = request.app.state.blob_store.get_json(str(lookup.step["output_hash"]))
    info = recorder.record_replayed_step(run_id, "LLM_CALL", step_name, body, output, lookup)
    if lookup.error is not None or output is None:
        return JSONResponse(
            {
                "error": {
                    "message": lookup.error or "recorded upstream call has no output",
                    "type": "checkpoint_replayed_error",
                }
            },
            status_code=502,
            headers=_settle_headers(info),
        )
    return JSONResponse(output, headers=_settle_headers(info))


async def _call_upstream(
    request: Request, body: dict[str, Any]
) -> tuple[httpx.Response | None, str | None, float]:
    started = time.perf_counter()
    upstream, forward_error = await _forward_upstream(request, body)
    return upstream, forward_error, (time.perf_counter() - started) * 1000.0


def _upstream_error(
    request: Request,
    run_id: str,
    step_name: str,
    body: dict[str, Any],
    forward_error: str | None,
    latency_ms: float,
    source: str,
) -> Response:
    recorder = request.app.state.recorder
    info = recorder.record_llm_step(
        run_id,
        step_name,
        body,
        None,
        latency_ms,
        error=forward_error,
        frozen_timestamp=request.headers.get(HEADER_FROZEN_TIMESTAMP) or real_now_iso(),
        source=source,
    )
    return JSONResponse(
        {"error": {"message": "upstream request failed", "type": "checkpoint_upstream_error"}},
        status_code=502,
        headers=_settle_headers(info),
    )


def _record_upstream_response(
    recorder: Any,
    run_id: str,
    step_name: str,
    body: dict[str, Any],
    upstream: httpx.Response,
    latency_ms: float,
    frozen_timestamp: str,
    source: str,
) -> Response:
    upstream_json: Any = None
    try:
        upstream_json = upstream.json()
    except ValueError:
        upstream_json = None
    error = None if upstream.is_success else f"upstream status {upstream.status_code}"
    if upstream_json is None and error is None:
        error = "upstream returned a non-JSON response"
    info = recorder.record_llm_step(
        run_id,
        step_name,
        body,
        upstream_json,
        latency_ms,
        error=error,
        frozen_timestamp=frozen_timestamp,
        source=source,
    )
    if upstream_json is None:
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/octet-stream"),
            headers=_settle_headers(info),
        )
    return JSONResponse(
        upstream_json, status_code=upstream.status_code, headers=_settle_headers(info)
    )
