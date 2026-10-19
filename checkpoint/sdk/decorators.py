"""``@tool`` and ``@step`` decorators backed by wrapt.

In RECORD mode a tool executes and its inputs/outputs are recorded. In REPLAY
mode ``pure`` and ``stub`` tools are served from the cassette; ``stub`` tools
never execute, and a miss is a divergence. ``live`` tools always execute and
are recorded with source LIVE.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast

import msgspec
import wrapt

from checkpoint.sdk import client as sdk_client
from checkpoint.sdk import determinism
from checkpoint.sdk.determinism import real_now_iso

F = TypeVar("F", bound=Callable[..., Any])

TOOL_MODES = ("pure", "stub", "live")


def _default_encode(value: Any) -> bytes:
    return msgspec.json.encode(value)


def _to_json_obj(raw: bytes | str) -> Any:
    if isinstance(raw, str):
        return json.loads(raw)
    return msgspec.json.decode(raw)


def tool(
    mode: str = "pure",
    name: str | None = None,
    serializer: Callable[[Any], bytes | str] | None = None,
    deserializer: Callable[[Any], Any] | None = None,
) -> Callable[[F], F]:
    """Record a tool call as a boundary step.

    ``mode`` is ``pure``, ``stub`` or ``live``. Inputs and outputs must be
    JSON serializable; pass ``serializer`` to override the default msgspec
    encoder. On replay ``deserializer`` receives the recorded JSON-decoded
    output and returns the value handed back to the agent.
    """
    if mode not in TOOL_MODES:
        raise ValueError(f"unknown tool mode: {mode}")
    encode = serializer or _default_encode

    def decorator(function: F) -> F:
        step_name = name or function.__name__

        @wrapt.decorator
        def wrapper(
            wrapped: Callable[..., Any],
            instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            state = sdk_client.run_state()
            input_obj = _to_json_obj(encode({"args": list(args), "kwargs": kwargs}))
            if state is None:
                return wrapped(*args, **kwargs)
            if state.mode in ("REPLAY", "FORK") and mode != "live":
                return _replay_tool(
                    wrapped, args, kwargs, step_name, mode, input_obj, encode, deserializer
                )
            return _execute_tool(
                wrapped, args, kwargs, step_name, mode, input_obj, encode, True
            )

        return cast(F, wrapper(function))

    return decorator


def _execute_tool(
    wrapped: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    step_name: str,
    mode: str,
    input_obj: Any,
    encode: Callable[[Any], bytes | str],
    begin_boundary: bool,
) -> Any:
    state = sdk_client.run_state()
    assert state is not None
    controller = determinism.controller()
    if begin_boundary and controller is not None:
        controller.begin_boundary()
    timestamp = real_now_iso()
    source = "RECORDED" if state.mode == "RECORD" else "LIVE"
    started = time.perf_counter()
    output_obj: Any = None
    error: str | None = None
    try:
        output_obj = wrapped(*args, **kwargs)
        return output_obj
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        latency_ms = (time.perf_counter() - started) * 1000.0
        info = sdk_client.record_step(
            {
                "run_id": state.run_id,
                "step_type": "TOOL_CALL",
                "name": step_name,
                "tool_mode": mode,
                "input": input_obj,
                "output": None if error else _to_json_obj(encode(output_obj)),
                "error": error,
                "latency_ms": latency_ms,
                "source": source,
                "frozen_timestamp": timestamp,
            }
        )
        sdk_client.settle(info)


def _replay_tool(
    wrapped: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    step_name: str,
    mode: str,
    input_obj: Any,
    encode: Callable[[Any], bytes | str],
    deserializer: Callable[[Any], Any] | None,
) -> Any:
    state = sdk_client.run_state()
    assert state is not None
    controller = determinism.controller()
    boundary = 0
    if controller is not None:
        boundary = controller.begin_boundary()
    response = sdk_client.lookup_step(
        {
            "run_id": state.run_id,
            "step_type": "TOOL_CALL",
            "name": step_name,
            "tool_mode": mode,
            "input": input_obj,
            "boundary_index": boundary,
        }
    )
    status = response.get("status")
    if status == "replayed":
        sdk_client.settle(response)
        output = response.get("output")
        return deserializer(output) if deserializer is not None else output
    if status == "replayed_error":
        sdk_client.settle(response)
        raise RuntimeError(f"recorded tool error: {response.get('error')}")
    if status == "live":
        return _execute_tool(wrapped, args, kwargs, step_name, mode, input_obj, encode, False)
    raise RuntimeError(f"unexpected replay response for tool {step_name}")


def step(name: str | None = None) -> Callable[[F], F]:
    """Label LLM calls made inside the decorated function."""

    def decorator(function: F) -> F:
        step_name = name or function.__name__

        @wrapt.decorator
        def wrapper(
            wrapped: Callable[..., Any],
            instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            with sdk_client.step_name_scope(step_name):
                return wrapped(*args, **kwargs)

        return cast(F, wrapper(function))

    return decorator
