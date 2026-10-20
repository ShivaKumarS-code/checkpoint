"""Recording and replaying proxy for OpenAI and Anthropic-style APIs.

RECORD forwards and records. REPLAY serves from the cassette. FORK serves the
parent's steps before ``forked_at_step`` and goes live from there, applying the
variant's ``llm_overrides`` to the live request.

Streaming responses are recorded as the exact upstream byte chunks. Replay
yields the same chunks in the same order, without contacting the upstream.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from checkpoint.daemon.experiments import (
    apply_llm_overrides,
    experiment_variant_overrides,
)
from checkpoint.daemon.recorder import (
    DEFAULT_LLM_STEP_NAME,
    StepInfo,
    StepReservation,
)
from checkpoint.daemon.streaming import decode_stream, encode_stream, stream_usage
from checkpoint.matching.replay import LookupResult, step_label
from checkpoint.matching.semantic import judge_prompt, parse_judge_response
from checkpoint.sdk.determinism import real_now_iso

router = APIRouter()

_HOP_BY_HOP = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}

HEADER_RUN_ID = "x-checkpoint-run-id"
HEADER_STEP_NAME = "x-checkpoint-step-name"
HEADER_FROZEN_TIMESTAMP = "x-checkpoint-frozen-timestamp"
HEADER_RNG_SEED = "x-checkpoint-rng-seed"
HEADER_BOUNDARY = "x-checkpoint-boundary"


@dataclass(frozen=True)
class Endpoint:
    upstream_path: str
    provider: str


OPENAI_ENDPOINT = Endpoint("/v1/chat/completions", "openai")
ANTHROPIC_ENDPOINT = Endpoint("/v1/messages", "anthropic")


def _forward_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        lower = key.lower()
        if lower in _HOP_BY_HOP or lower.startswith("x-checkpoint-"):
            continue
        headers[key] = value
    return headers


def _settle_headers(info: StepInfo | StepReservation) -> dict[str, str]:
    return {
        HEADER_FROZEN_TIMESTAMP: info.frozen_timestamp,
        HEADER_RNG_SEED: str(info.rng_seed),
    }


def _stream_headers(info: StepInfo | StepReservation) -> dict[str, str]:
    return {
        **_settle_headers(info),
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
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
    request: Request,
    body: dict[str, Any],
    endpoint: Endpoint,
) -> tuple[httpx.Response | None, str | None]:
    upstream_url = f"{request.app.state.upstream_url}{endpoint.upstream_path}"
    try:
        upstream = await request.app.state.http.post(
            upstream_url, json=body, headers=_forward_headers(request)
        )
    except httpx.HTTPError as exc:
        return None, f"upstream request failed: {type(exc).__name__}"
    return upstream, None


async def _open_upstream_stream(
    request: Request,
    body: dict[str, Any],
    endpoint: Endpoint,
) -> tuple[Any | None, httpx.Response | None, str | None]:
    """Open an upstream stream and return (context manager, response, error)."""
    upstream_url = f"{request.app.state.upstream_url}{endpoint.upstream_path}"
    manager = request.app.state.http.stream(
        "POST", upstream_url, json=body, headers=_forward_headers(request)
    )
    try:
        upstream = await manager.__aenter__()
    except httpx.HTTPError as exc:
        with contextlib.suppress(Exception):
            await manager.__aexit__(None, None, None)
        return None, None, f"upstream request failed: {type(exc).__name__}"
    return manager, upstream, None


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    return await _handle_llm(request, OPENAI_ENDPOINT)


@router.post("/openai/v1/chat/completions")
async def groq_chat_completions(request: Request) -> Response:
    """Groq's SDK prefixes its OpenAI-compatible route with ``/openai``."""
    return await _handle_llm(request, OPENAI_ENDPOINT)


@router.post("/v1/messages")
async def messages(request: Request) -> Response:
    return await _handle_llm(request, ANTHROPIC_ENDPOINT)


async def _handle_llm(request: Request, endpoint: Endpoint) -> Response:
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

    mode = str(run["mode"])
    if mode == "REPLAY":
        return await _replay_llm(request, run_id, body, endpoint)
    if mode == "FORK":
        return await _fork_llm(request, run_id, run, body, endpoint)
    if mode != "RECORD":
        return JSONResponse(
            {"error": {"message": f"run mode {mode} is not supported yet"}},
            status_code=501,
        )
    return await _record_llm(request, run_id, body, endpoint, "RECORDED")


async def _record_llm(
    request: Request,
    run_id: str,
    body: dict[str, Any],
    endpoint: Endpoint,
    source: str,
) -> Response:
    if body.get("stream"):
        return await _record_stream(request, run_id, body, endpoint, source)
    return await _record_nonstream(request, run_id, body, endpoint, source)


async def _record_nonstream(
    request: Request,
    run_id: str,
    body: dict[str, Any],
    endpoint: Endpoint,
    source: str,
) -> Response:
    recorder = request.app.state.recorder
    step_name = request.headers.get(HEADER_STEP_NAME) or DEFAULT_LLM_STEP_NAME
    frozen_timestamp = request.headers.get(HEADER_FROZEN_TIMESTAMP) or real_now_iso()
    started = time.perf_counter()
    upstream, forward_error = await _forward_upstream(request, body, endpoint)
    latency_ms = (time.perf_counter() - started) * 1000.0
    if upstream is None:
        return _upstream_error(
            request, run_id, step_name, body, forward_error, latency_ms, source
        )
    return _record_upstream_response(
        recorder, run_id, step_name, body, upstream, latency_ms, frozen_timestamp, source
    )


async def _record_stream(
    request: Request,
    run_id: str,
    body: dict[str, Any],
    endpoint: Endpoint,
    source: str,
) -> Response:
    recorder = request.app.state.recorder
    step_name = request.headers.get(HEADER_STEP_NAME) or DEFAULT_LLM_STEP_NAME
    frozen_timestamp = request.headers.get(HEADER_FROZEN_TIMESTAMP) or real_now_iso()
    started = time.perf_counter()
    manager, upstream, forward_error = await _open_upstream_stream(request, body, endpoint)
    if manager is None or upstream is None:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return _upstream_error(
            request, run_id, step_name, body, forward_error, latency_ms, source
        )

    content_type = upstream.headers.get("content-type", "text/event-stream")
    if upstream.status_code >= 400:
        content = await upstream.aread()
        await manager.__aexit__(None, None, None)
        latency_ms = (time.perf_counter() - started) * 1000.0
        info = recorder.record_llm_step(
            run_id,
            step_name,
            body,
            encode_stream(
                [content],
                content_type=content_type,
                status_code=upstream.status_code,
            ),
            latency_ms,
            error=f"upstream status {upstream.status_code}",
            frozen_timestamp=frozen_timestamp,
            source=source,
        )
        return Response(
            content=content,
            status_code=upstream.status_code,
            media_type=content_type,
            headers=_settle_headers(info),
        )

    reservation = recorder.reserve_stream_step(run_id, step_name, frozen_timestamp)

    async def generate() -> AsyncIterator[bytes]:
        chunks: list[bytes] = []
        error: str | None = None
        try:
            async for chunk in upstream.aiter_raw():
                if chunk:
                    chunks.append(chunk)
                    yield chunk
        except asyncio.CancelledError:
            error = "client disconnected while streaming"
            raise
        except Exception as exc:  # pragma: no cover - transport failure path
            error = f"upstream stream failed: {type(exc).__name__}"
        finally:
            with contextlib.suppress(Exception):
                await manager.__aexit__(None, None, None)
            usage = stream_usage(chunks)
            payload = encode_stream(
                chunks,
                content_type=content_type,
                status_code=upstream.status_code,
                usage=usage,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            with contextlib.suppress(Exception):
                recorder.record_reserved_llm_step(
                    run_id,
                    step_name,
                    body,
                    payload,
                    latency_ms,
                    reservation,
                    error=error,
                    source=source,
                )

    return StreamingResponse(
        generate(),
        status_code=upstream.status_code,
        media_type=content_type,
        headers=_stream_headers(reservation),
    )


async def _replay_llm(
    request: Request,
    run_id: str,
    body: dict[str, Any],
    endpoint: Endpoint,
) -> Response:
    recorder = request.app.state.recorder
    step_name = request.headers.get(HEADER_STEP_NAME) or DEFAULT_LLM_STEP_NAME
    lookup = recorder.lookup_llm(run_id, step_name, body)

    if lookup.found and lookup.matched:
        return _serve_replayed(request, run_id, step_name, body, lookup)

    semantic = await _semantic_lookup(request, run_id, step_name, body)
    if semantic is not None:
        return _serve_replayed(request, run_id, step_name, body, semantic)

    if request.app.state.config.replay.llm_miss_policy == "live":
        return await _record_llm(request, run_id, body, endpoint, "LIVE")

    label = step_label("LLM_CALL", step_name, lookup.occurrence)
    diff = lookup.diff or f"no recorded {label} in the parent run"
    recorder.mark_diverged(run_id, lookup.step_index, diff)
    return _diverged(f"replay diverged at {label}", diff, lookup.step_index)


async def _fork_llm(
    request: Request,
    run_id: str,
    run: Any,
    body: dict[str, Any],
    endpoint: Endpoint,
) -> Response:
    """Serve parent steps before N, then call the upstream with variant overrides."""
    recorder = request.app.state.recorder
    step_name = request.headers.get(HEADER_STEP_NAME) or DEFAULT_LLM_STEP_NAME
    lookup = recorder.lookup_llm(run_id, step_name, body)
    fork_index = int(run["forked_at_step"] or 0)
    boundary = _boundary(request)

    if lookup.found and lookup.step_index is not None and lookup.step_index < fork_index:
        if lookup.matched:
            return _serve_replayed(request, run_id, step_name, body, lookup)
        semantic = await _semantic_lookup(
            request, run_id, step_name, body, max_step_index=fork_index
        )
        if semantic is not None:
            return _serve_replayed(request, run_id, step_name, body, semantic)
        label = step_label("LLM_CALL", step_name, lookup.occurrence)
        diff = lookup.diff or f"no recorded {label} in the parent run"
        recorder.mark_diverged(run_id, lookup.step_index, diff)
        return _diverged(
            f"code changed before the fork point: {label}", diff, lookup.step_index
        )

    if not lookup.found and boundary is not None and boundary < fork_index:
        semantic = await _semantic_lookup(
            request, run_id, step_name, body, max_step_index=fork_index
        )
        if semantic is not None:
            return _serve_replayed(request, run_id, step_name, body, semantic)
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
    return await _record_llm(request, run_id, effective, endpoint, "LIVE")


async def _semantic_lookup(
    request: Request,
    run_id: str,
    step_name: str,
    body: dict[str, Any],
    *,
    max_step_index: int | None = None,
) -> LookupResult | None:
    semantic = request.app.state.config.semantic
    if not semantic.enabled or step_name not in semantic.step_names:
        return None
    recorder = request.app.state.recorder
    lookup = recorder.lookup_semantic(
        run_id,
        step_name,
        body,
        threshold=semantic.threshold,
        step_names=semantic.step_names,
        max_step_index=max_step_index,
    )
    if lookup is None:
        return None
    if semantic.judge:
        confirmed = await _semantic_judge(
            request, body, lookup.recorded_input, semantic.judge_model
        )
        if not confirmed:
            return None
    return lookup


async def _semantic_judge(
    request: Request,
    incoming: dict[str, Any],
    recorded: Any,
    model: str,
) -> bool:
    if not isinstance(recorded, dict):
        return False
    prompt = judge_prompt(incoming, recorded)
    headers = _forward_headers(request)
    judge_key = os.environ.get("CHECKPOINT_JUDGE_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    if judge_key:
        headers["Authorization"] = f"Bearer {judge_key}"
    try:
        response = await request.app.state.http.post(
            f"{request.app.state.upstream_url.rstrip('/')}/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "temperature": 0.0,
                "max_tokens": 200,
                "messages": [
                    {
                        "role": "system",
                        "content": "Decide equivalence and reply with JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return False
    choices = payload.get("choices") if isinstance(payload, dict) else None
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        return False
    confirmed, _ = parse_judge_response(content)
    return confirmed


def _serve_replayed(
    request: Request,
    run_id: str,
    step_name: str,
    body: dict[str, Any],
    lookup: LookupResult,
) -> Response:
    recorder = request.app.state.recorder
    output: Any = None
    if lookup.step is not None and lookup.step["output_hash"] is not None:
        output = request.app.state.blob_store.get_json(str(lookup.step["output_hash"]))
    source = "SEMANTIC" if lookup.semantic else "REPLAYED"
    info = recorder.record_replayed_step(
        run_id,
        "LLM_CALL",
        step_name,
        body,
        output,
        lookup,
        reuse_parent_index=not lookup.semantic,
        source=source,
    )
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
    envelope = decode_stream(output)
    if envelope is not None:
        chunks, content_type, status_code, _usage = envelope
        return StreamingResponse(
            _iter_chunks(chunks),
            status_code=status_code,
            media_type=content_type,
            headers=_stream_headers(info),
        )
    return JSONResponse(output, headers=_settle_headers(info))


async def _iter_chunks(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


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
