"""``@tool`` and ``@step`` decorators backed by wrapt."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast

import msgspec
import wrapt

from checkpoint.sdk import client as sdk_client

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
    encoder. ``deserializer`` is used by replay (Phase 2).
    """
    del deserializer
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
            if state is None or state.mode != "RECORD":
                return wrapped(*args, **kwargs)
            input_obj = _to_json_obj(encode({"args": list(args), "kwargs": kwargs}))
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
                sdk_client.record_step(
                    {
                        "run_id": state.run_id,
                        "step_type": "TOOL_CALL",
                        "name": step_name,
                        "tool_mode": mode,
                        "input": input_obj,
                        "output": None if error else _to_json_obj(encode(output_obj)),
                        "error": error,
                        "latency_ms": latency_ms,
                    }
                )

        return cast(F, wrapper(function))

    return decorator


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
