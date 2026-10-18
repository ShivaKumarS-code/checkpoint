"""Daemon RPC client plus the httpx patch that tags proxy requests.

``checkpoint.init()`` patches ``httpx.Client.send`` and
``httpx.AsyncClient.send`` so that requests aimed at the daemon carry
``x-checkpoint-run-id`` and, inside a ``@checkpoint.step``, the step name.
The OpenAI and Anthropic SDKs use httpx, so this works without wrapping them.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
import wrapt

ENV_DAEMON_URL = "CHECKPOINT_DAEMON_URL"
ENV_RUN_ID = "CHECKPOINT_RUN_ID"
ENV_MODE = "CHECKPOINT_MODE"

HEADER_RUN_ID = "x-checkpoint-run-id"
HEADER_STEP_NAME = "x-checkpoint-step-name"


@dataclass
class RunState:
    run_id: str
    mode: str
    daemon_url: str


_state: RunState | None = None
_step_name: ContextVar[str | None] = ContextVar("checkpoint_step_name", default=None)
_patched = False


def run_state() -> RunState | None:
    return _state


def reset() -> None:
    """Forget the current run (used by tests and process shutdown)."""
    global _state
    _state = None


def daemon_url_from_env() -> str | None:
    url = os.environ.get(ENV_DAEMON_URL)
    return url.rstrip("/") if url else None


def current_step_name() -> str | None:
    return _step_name.get()


@contextlib.contextmanager
def step_name_scope(name: str) -> Iterator[None]:
    token = _step_name.set(name)
    try:
        yield
    finally:
        _step_name.reset(token)


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = daemon_url_from_env()
    if url is None:
        raise RuntimeError(f"{ENV_DAEMON_URL} is not set")
    response = httpx.post(f"{url}{path}", json=payload, timeout=30.0)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"unexpected response from daemon at {path}")
    return data


def start_run(name: str, mode: str) -> dict[str, Any]:
    return _post("/api/sdk/run/start", {"name": name, "mode": mode})


def record_step(payload: dict[str, Any]) -> dict[str, Any]:
    return _post("/api/sdk/step", payload)


def init(run_name: str = "run") -> str | None:
    """Bind this process to a daemon run.

    Reads ``CHECKPOINT_DAEMON_URL``, ``CHECKPOINT_RUN_ID`` (optional) and
    ``CHECKPOINT_MODE``. No-op returning ``None`` when not under a daemon.
    """
    global _state
    url = daemon_url_from_env()
    if url is None:
        _state = None
        return None
    mode = os.environ.get(ENV_MODE, "record").strip().upper() or "RECORD"
    run_id = os.environ.get(ENV_RUN_ID)
    if not run_id:
        created = start_run(run_name, mode)
        run_id = str(created["run_id"])
    _state = RunState(run_id=run_id, mode=mode, daemon_url=url)
    install_httpx_patch()
    return run_id


def _tag_request(request: httpx.Request) -> None:
    state = _state
    if state is None:
        return
    if urlsplit(str(request.url)).netloc != urlsplit(state.daemon_url).netloc:
        return
    if HEADER_RUN_ID not in request.headers:
        request.headers[HEADER_RUN_ID] = state.run_id
    step_name = current_step_name()
    if step_name and HEADER_STEP_NAME not in request.headers:
        request.headers[HEADER_STEP_NAME] = step_name


def _request_from(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if "request" in kwargs:
        return kwargs["request"]
    return args[0] if args else None


def _sync_send(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    request = _request_from(args, kwargs)
    if isinstance(request, httpx.Request):
        _tag_request(request)
    return wrapped(*args, **kwargs)


async def _async_send(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    request = _request_from(args, kwargs)
    if isinstance(request, httpx.Request):
        _tag_request(request)
    return await wrapped(*args, **kwargs)


def install_httpx_patch() -> None:
    global _patched
    if _patched:
        return
    wrapt.wrap_function_wrapper(httpx.Client, "send", _sync_send)
    wrapt.wrap_function_wrapper(httpx.AsyncClient, "send", _async_send)
    _patched = True
