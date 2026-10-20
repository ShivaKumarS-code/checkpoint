"""SDK RPC endpoints: run registration, replay lookup and step recording."""

from __future__ import annotations

from typing import Any, Literal

import msgspec
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from checkpoint.matching.replay import step_label
from checkpoint.storage.models import RunMode, ToolMode

router = APIRouter()


class StartRunRequest(msgspec.Struct):
    name: str = "run"
    mode: RunMode = "RECORD"
    parent_run_id: str | None = None
    run_seed: int | None = None
    forked_at_step: int | None = None
    experiment_id: str | None = None
    variant_name: str | None = None


class StepRequest(msgspec.Struct):
    run_id: str
    name: str
    step_type: Literal["TOOL_CALL", "CHECKPOINT"] = "TOOL_CALL"
    tool_mode: ToolMode | None = None
    input: Any = None
    output: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    source: Literal["RECORDED", "LIVE"] = "RECORDED"
    frozen_timestamp: str | None = None


class LookupRequest(msgspec.Struct):
    run_id: str
    name: str
    step_type: Literal["TOOL_CALL", "CHECKPOINT"] = "TOOL_CALL"
    tool_mode: ToolMode | None = None
    input: Any = None
    boundary_index: int | None = None


def _bad_request(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": {"message": message}}, status_code=status)


def _settle_payload(info: Any) -> dict[str, Any]:
    return {
        "step_index": info.step_index,
        "occurrence": info.occurrence,
        "frozen_timestamp": info.frozen_timestamp,
        "rng_seed": info.rng_seed,
    }


@router.post("/api/sdk/run/start")
async def start_run(request: Request) -> JSONResponse:
    try:
        payload = msgspec.json.decode(await request.body(), type=StartRunRequest)
    except msgspec.ValidationError as exc:
        return _bad_request(str(exc))
    created = request.app.state.recorder.create_run(
        payload.name,
        payload.mode,
        parent_run_id=payload.parent_run_id,
        forked_at_step=payload.forked_at_step,
        variant_name=payload.variant_name,
        experiment_id=payload.experiment_id,
        run_seed=payload.run_seed,
    )
    return JSONResponse(created)


@router.get("/api/sdk/run/{run_id}")
async def run_info(run_id: str, request: Request) -> JSONResponse:
    recorder = request.app.state.recorder
    run = recorder.get_run(run_id)
    if run is None:
        return _bad_request(f"unknown run: {run_id}", status=404)
    root = recorder.root_run(run_id) or run
    return JSONResponse(
        {
            "run_id": run["run_id"],
            "name": run["name"],
            "mode": run["mode"],
            "run_seed": int(run["run_seed"]),
            "parent_run_id": run["parent_run_id"],
            "forked_at_step": run["forked_at_step"],
            "experiment_id": run["experiment_id"],
            "variant_name": run["variant_name"],
            "clock_start": float(root["created_at"]),
            "freeze_clock": bool(request.app.state.config.freeze_clock),
        }
    )


@router.post("/api/sdk/lookup")
async def lookup_step(request: Request) -> JSONResponse:
    try:
        payload = msgspec.json.decode(await request.body(), type=LookupRequest)
    except msgspec.ValidationError as exc:
        return _bad_request(str(exc))
    recorder = request.app.state.recorder
    run = recorder.get_run(payload.run_id)
    if run is None:
        return _bad_request(f"unknown run: {payload.run_id}", status=404)
    if run["mode"] not in ("REPLAY", "FORK"):
        return _bad_request(f"run mode {run['mode']} cannot replay steps")

    if payload.step_type == "CHECKPOINT":
        lookup = recorder.lookup_checkpoint(payload.run_id, payload.name)
    else:
        lookup = recorder.lookup_tool(payload.run_id, payload.name, payload.input)

    if run["mode"] == "FORK":
        return _fork_lookup(request, run, payload, lookup)
    return _replay_lookup(request, payload, lookup)


def _replay_lookup(
    request: Request, payload: LookupRequest, lookup: Any
) -> JSONResponse:
    recorder = request.app.state.recorder
    if lookup.found and lookup.matched:
        return _serve_replayed(request, payload, lookup)

    policy = (
        request.app.state.config.replay.tool_miss_policy
        if payload.step_type == "TOOL_CALL"
        else request.app.state.config.replay.checkpoint_miss_policy
    )
    if payload.step_type == "TOOL_CALL" and payload.tool_mode == "stub":
        policy = "strict"

    if policy == "live":
        return JSONResponse({"status": "live", "diff": lookup.diff})

    return _diverge(recorder, payload, lookup, "replay diverged")


def _fork_lookup(
    request: Request, run: Any, payload: LookupRequest, lookup: Any
) -> JSONResponse:
    """Fork rules: parent steps before N replay, later calls go live."""
    recorder = request.app.state.recorder
    fork_at = run["forked_at_step"]
    fork_index = int(fork_at) if fork_at is not None else 0
    boundary = payload.boundary_index
    before_fork = boundary is not None and boundary < fork_index
    parent_index = lookup.step_index

    if lookup.found and parent_index is not None:
        if parent_index < fork_index:
            if lookup.matched:
                return _serve_replayed(request, payload, lookup)
            return _diverge(recorder, payload, lookup, "code changed before the fork point")
        if payload.step_type == "TOOL_CALL" and payload.tool_mode == "stub":
            if lookup.matched:
                return _serve_replayed(request, payload, lookup, reuse_parent_index=False)
            return _bad_request(
                f"stub tool {payload.name!r} input does not match the recording and "
                "stubs never execute during a fork"
            )
        return JSONResponse({"status": "live", "diff": lookup.diff})

    if payload.step_type == "TOOL_CALL" and payload.tool_mode == "stub":
        return _bad_request(
            f"stub tool {payload.name!r} has no recorded output and never executes "
            "during a fork"
        )
    if before_fork:
        return _diverge(recorder, payload, lookup, "missing step before the fork point")
    return JSONResponse({"status": "live", "diff": lookup.diff})


def _serve_replayed(
    request: Request,
    payload: LookupRequest,
    lookup: Any,
    reuse_parent_index: bool = True,
) -> JSONResponse:
    recorder = request.app.state.recorder
    output = None
    if lookup.step is not None and lookup.step["output_hash"] is not None:
        output = request.app.state.blob_store.get_json(str(lookup.step["output_hash"]))
    info = recorder.record_replayed_step(
        payload.run_id,
        payload.step_type,
        payload.name,
        payload.input,
        output,
        lookup,
        tool_mode=payload.tool_mode,
        reuse_parent_index=reuse_parent_index,
    )
    if lookup.error is not None:
        return JSONResponse(
            {"status": "replayed_error", "error": lookup.error, **_settle_payload(info)}
        )
    return JSONResponse({"status": "replayed", "output": output, **_settle_payload(info)})


def _diverge(
    recorder: Any, payload: LookupRequest, lookup: Any, prefix: str
) -> JSONResponse:
    label = step_label(payload.step_type, payload.name, lookup.occurrence)
    diff = lookup.diff or f"no recorded {label} in the parent run"
    recorder.mark_diverged(payload.run_id, lookup.step_index, diff)
    return JSONResponse(
        {
            "error": {
                "message": f"{prefix}: {label}",
                "type": "checkpoint_divergence",
                "diff": diff,
            },
            "step_index": lookup.step_index,
            "occurrence": lookup.occurrence,
        },
        status_code=409,
    )


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
    if run["mode"] not in ("RECORD", "REPLAY", "FORK"):
        return JSONResponse(
            {"error": {"message": f"run mode {run['mode']} does not record steps yet"}},
            status_code=501,
        )
    if run["mode"] in ("REPLAY", "FORK") and payload.source != "LIVE":
        return _bad_request("replayed steps are recorded by the daemon lookup")
    source = "LIVE" if payload.source == "LIVE" else "RECORDED"
    if payload.step_type == "CHECKPOINT":
        info = recorder.record_checkpoint(
            payload.run_id,
            payload.name,
            payload.output,
            frozen_timestamp=payload.frozen_timestamp,
            source=source,
        )
    else:
        info = recorder.record_tool_step(
            payload.run_id,
            payload.name,
            payload.input,
            payload.output,
            tool_mode=payload.tool_mode,
            error=payload.error,
            latency_ms=payload.latency_ms,
            frozen_timestamp=payload.frozen_timestamp,
            source=source,
        )
    response: dict[str, Any] = _settle_payload(info)
    response["source"] = source
    return JSONResponse(response)
