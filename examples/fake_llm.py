"""Deterministic OpenAI-compatible fake provider for tests and the demo agent.

Every response is a pure function of the request body: identical requests get
identical responses, and no API key is needed anywhere.

Demo-agent conventions understood by the fake:

- ``{"task": "plan", ...}``: extracts the first ``ORD-\\d+`` from the ticket.
- ``{"task": "decide", ...}``: approves iff ``policy.refundable`` is true,
  unless the system prompt mentions ``prior_refund_id`` and the order carries
  one, in which case it denies. This is what a fork variant can change.
- ``{"task": "compose", ...}``: writes a short customer reply.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

_ORDER_RE = re.compile(r"ORD-\d+")
_FALLBACK_TEXT = "This is a deterministic fake completion."


class _Stats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: list[dict[str, Any]] = []

    def record(self, body: Mapping[str, Any]) -> None:
        with self._lock:
            self._calls.append(dict(body))

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._calls)


_stats = _Stats()


def call_count() -> int:
    return _stats.count()


def reset() -> None:
    _stats.reset()


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return json.dumps(content, sort_keys=True, default=str)


def _messages(body: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = body.get("messages")
    if not isinstance(raw, list):
        return []
    return [message for message in raw if isinstance(message, Mapping)]


def _last_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _message_text(message)
    return ""


def _system_text(messages: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        _message_text(message) for message in messages if message.get("role") == "system"
    )


def _completion_text(user_text: str, system_text: str) -> str:
    try:
        payload: Any = json.loads(user_text)
    except ValueError:
        return _FALLBACK_TEXT
    if not isinstance(payload, Mapping):
        return _FALLBACK_TEXT

    task = payload.get("task")
    if task == "plan":
        match = _ORDER_RE.search(str(payload.get("ticket", "")))
        order_id = match.group(0) if match else None
        return json.dumps({"order_id": order_id, "action": "refund" if order_id else "none"})

    if task == "decide":
        order = payload.get("order")
        policy = payload.get("policy")
        order = order if isinstance(order, Mapping) else {}
        policy = policy if isinstance(policy, Mapping) else {}
        strict = "prior_refund_id" in system_text
        if strict and order.get("prior_refund_id"):
            decision, reason = "deny", "order already has a prior refund"
        elif policy.get("refundable"):
            decision, reason = "approve", "order is refundable"
        else:
            decision, reason = "deny", "order is not refundable"
        amount = policy.get("amount") if decision == "approve" else None
        return json.dumps({"decision": decision, "amount": amount, "reason": reason})

    if task == "compose":
        decision = payload.get("decision")
        amount = payload.get("amount")
        refund_id = payload.get("refund_id")
        if decision == "approve" and isinstance(refund_id, str):
            if isinstance(amount, (int, float)):
                return f"Refund {refund_id} of {amount:.2f} has been issued."
            return f"Refund {refund_id} has been issued."
        if decision == "approve":
            return "Refund approved."
        return "We reviewed your request and cannot approve a refund for this order."

    return _FALLBACK_TEXT


def _request_digest(body: Mapping[str, Any]) -> str:
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def generate_completion(body: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic OpenAI-compatible chat completion for a request body."""
    messages = _messages(body)
    user_text = _last_user_text(messages)
    system_text = _system_text(messages)
    content = _completion_text(user_text, system_text)
    prompt_tokens = max(1, (len(user_text) + len(system_text)) // 4)
    completion_tokens = max(1, len(content) // 4)
    return {
        "id": f"chatcmpl-{_request_digest(body)}",
        "object": "chat.completion",
        "created": 0,
        "model": str(body.get("model", "fake-llm")),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def create_app() -> FastAPI:
    app = FastAPI(title="checkpoint fake LLM", docs_url=None, redoc_url=None)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": {"message": "invalid JSON body"}}, status_code=400)
        if not isinstance(body, Mapping) or not isinstance(body.get("messages"), list):
            return JSONResponse(
                {"error": {"message": "messages is required"}}, status_code=400
            )
        _stats.record(body)
        return JSONResponse(generate_completion(body))

    @app.get("/__control/stats")
    async def stats() -> JSONResponse:
        return JSONResponse({"calls": _stats.count()})

    @app.post("/__control/reset")
    async def reset_endpoint() -> JSONResponse:
        _stats.reset()
        return JSONResponse({"calls": _stats.count()})

    return app


app = create_app()


class FakeLLMServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._host = host
        self._port = port
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def start(self) -> FakeLLMServer:
        config = uvicorn.Config(app, host=self._host, port=self._port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="fake-llm", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10.0
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError("fake LLM server did not start")
            time.sleep(0.01)
        sockname = self._server.servers[0].sockets[0].getsockname()
        self._port = int(sockname[1])
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        self._server = None


def serve(host: str = "127.0.0.1", port: int = 0) -> FakeLLMServer:
    """Start a fake LLM server in a background thread."""
    return FakeLLMServer(host, port).start()


def main() -> None:
    port = int(os.environ.get("FAKE_LLM_PORT", "4141"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
