"""Daemon RPC client, determinism bootstrap, and the httpx proxy tagging patch.

``checkpoint.init()`` fetches the run's seed and clock start from the daemon,
then installs deterministic time/UUID/urandom/random handling. It also patches
``httpx.Client.send`` and ``httpx.AsyncClient.send`` so requests aimed at the
daemon's ``/v1`` proxy carry ``x-checkpoint-run-id``, the step name and the
boundary timestamp. The OpenAI and Anthropic SDKs use httpx, so this works
without wrapping them.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
import wrapt

from checkpoint.sdk import determinism
from checkpoint.sdk.determinism import SeedHook, real_now_iso

ENV_DAEMON_URL = "CHECKPOINT_DAEMON_URL"
ENV_RUN_ID = "CHECKPOINT_RUN_ID"
ENV_MODE = "CHECKPOINT_MODE"

HEADER_RUN_ID = "x-checkpoint-run-id"
HEADER_STEP_NAME = "x-checkpoint-step-name"
HEADER_FROZEN_TIMESTAMP = "x-checkpoint-frozen-timestamp"
HEADER_RNG_SEED = "x-checkpoint-rng-seed"
HEADER_BOUNDARY = "x-checkpoint-boundary"


class DivergenceError(RuntimeError):
    """Raised when replay finds a boundary call that does not match the cassette."""

    def __init__(self, message: str, diff: str | None = None) -> None:
        super().__init__(message)
        self.diff = diff


@dataclass
class RunState:
    run_id: str
    mode: str
    daemon_url: str
    run_seed: int
    clock_start: float
    freeze_clock: bool
    parent_run_id: str | None
    forked_at_step: int | None = None


_state: RunState | None = None
_step_name: ContextVar[str | None] = ContextVar("checkpoint_step_name", default=None)
_patched = False


def run_state() -> RunState | None:
    return _state


def reset() -> None:
    """Forget the current run (used by tests and process shutdown)."""
    global _state
    _state = None
    determinism.stop()


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


def _decode(response: httpx.Response, path: str) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        data = None
    if response.status_code >= 400:
        error = data.get("error") if isinstance(data, dict) else None
        message = (
            str(error.get("message"))
            if isinstance(error, dict) and error.get("message")
            else f"daemon returned HTTP {response.status_code}"
        )
        if isinstance(error, dict) and error.get("type") == "checkpoint_divergence":
            diff = error.get("diff")
            raise DivergenceError(message, diff if isinstance(diff, str) else None)
        raise RuntimeError(f"checkpoint daemon request to {path} failed: {message}")
    if not isinstance(data, dict):
        raise RuntimeError(f"unexpected response from daemon at {path}")
    return data


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = daemon_url_from_env()
    if url is None:
        raise RuntimeError(f"{ENV_DAEMON_URL} is not set")
    response = httpx.post(f"{url}{path}", json=payload, timeout=30.0)
    return _decode(response, path)


def _get(path: str) -> dict[str, Any]:
    url = daemon_url_from_env()
    if url is None:
        raise RuntimeError(f"{ENV_DAEMON_URL} is not set")
    response = httpx.get(f"{url}{path}", timeout=30.0)
    return _decode(response, path)


def start_run(name: str, mode: str) -> dict[str, Any]:
    return _post("/api/sdk/run/start", {"name": name, "mode": mode})


def get_run_info(run_id: str) -> dict[str, Any]:
    return _get(f"/api/sdk/run/{run_id}")


def record_step(payload: dict[str, Any]) -> dict[str, Any]:
    return _post("/api/sdk/step", payload)


def lookup_step(payload: dict[str, Any]) -> dict[str, Any]:
    return _post("/api/sdk/lookup", payload)


def settle(info: dict[str, Any]) -> None:
    """Advance the frozen clock to a boundary's recorded timestamp."""
    controller = determinism.controller()
    if controller is None:
        return
    timestamp = info.get("frozen_timestamp")
    if isinstance(timestamp, str):
        controller.advance_to(timestamp)
    seed = info.get("rng_seed")
    if isinstance(seed, int):
        controller.seed_with(seed)


def init(
    run_name: str = "run", seed_hooks: Iterable[SeedHook] = ()
) -> str | None:
    """Bind this process to a daemon run and install determinism.

    Reads ``CHECKPOINT_DAEMON_URL``, ``CHECKPOINT_RUN_ID`` (optional) and
    ``CHECKPOINT_MODE``. No-op returning ``None`` when not under a daemon.
    """
    global _state
    url = daemon_url_from_env()
    if url is None:
        _state = None
        determinism.stop()
        return None
    mode = os.environ.get(ENV_MODE, "record").strip().upper() or "RECORD"
    run_id = os.environ.get(ENV_RUN_ID)
    if not run_id:
        created = start_run(run_name, mode)
        run_id = str(created["run_id"])
    info = get_run_info(run_id)
    _state = RunState(
        run_id=run_id,
        mode=str(info["mode"]),
        daemon_url=url,
        run_seed=int(info["run_seed"]),
        clock_start=float(info["clock_start"]),
        freeze_clock=bool(info["freeze_clock"]),
        parent_run_id=info.get("parent_run_id"),
        forked_at_step=info.get("forked_at_step"),
    )
    determinism.install(
        run_seed=_state.run_seed,
        clock_start=_state.clock_start,
        freeze_clock=_state.freeze_clock,
        seed_hooks=seed_hooks,
    )
    install_httpx_patch()
    return run_id


def _tag_request(request: httpx.Request) -> None:
    state = _state
    if state is None:
        return
    parsed = urlsplit(str(request.url))
    if parsed.netloc != urlsplit(state.daemon_url).netloc:
        return
    if not (
        parsed.path.startswith("/v1/")
        or parsed.path.startswith("/openai/v1/")
    ):
        return
    if HEADER_RUN_ID not in request.headers:
        request.headers[HEADER_RUN_ID] = state.run_id
    step_name = current_step_name()
    if step_name and HEADER_STEP_NAME not in request.headers:
        request.headers[HEADER_STEP_NAME] = step_name
    controller = determinism.controller()
    if controller is not None and HEADER_BOUNDARY not in request.headers:
        request.headers[HEADER_BOUNDARY] = str(controller.begin_boundary())
    if state.mode in ("RECORD", "FORK") and HEADER_FROZEN_TIMESTAMP not in request.headers:
        request.headers[HEADER_FROZEN_TIMESTAMP] = real_now_iso()


def _settle_response(response: Any) -> None:
    if not isinstance(response, httpx.Response):
        return
    controller = determinism.controller()
    if controller is None:
        return
    timestamp = response.headers.get(HEADER_FROZEN_TIMESTAMP)
    if timestamp:
        controller.advance_to(timestamp)
    seed = response.headers.get(HEADER_RNG_SEED)
    if seed is not None:
        try:
            controller.seed_with(int(seed))
        except ValueError:
            pass


def _request_from(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if "request" in kwargs:
        return kwargs["request"]
    return args[0] if args else None


def _sync_send(
    wrapped: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    request = _request_from(args, kwargs)
    if isinstance(request, httpx.Request):
        _tag_request(request)
    response = wrapped(*args, **kwargs)
    _settle_response(response)
    return response


async def _async_send(
    wrapped: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    request = _request_from(args, kwargs)
    if isinstance(request, httpx.Request):
        _tag_request(request)
    response = await wrapped(*args, **kwargs)
    _settle_response(response)
    return response


def install_httpx_patch() -> None:
    global _patched
    if _patched:
        return
    wrapt.wrap_function_wrapper(httpx.Client, "send", _sync_send)
    wrapt.wrap_function_wrapper(httpx.AsyncClient, "send", _async_send)
    _patched = True
