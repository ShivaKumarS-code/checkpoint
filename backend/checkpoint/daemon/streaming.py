"""Helpers for recording and replaying SSE response chunks.

The proxy stores the exact byte chunks returned by the upstream in the step's
output blob. Chunks are base64 encoded so arbitrary bytes round-trip through
the JSON blob store, and replay yields the same sequence and boundaries.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from typing import Any

STREAM_MARKER = "__checkpoint_stream__"


def encode_stream(
    chunks: Sequence[bytes],
    *,
    content_type: str,
    status_code: int,
    usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON envelope stored as a streaming LLM step's output."""
    return {
        STREAM_MARKER: True,
        "content_type": content_type,
        "status_code": status_code,
        "chunks": [base64.b64encode(chunk).decode("ascii") for chunk in chunks],
        "usage": dict(usage) if usage else None,
    }


def decode_stream(payload: Any) -> tuple[list[bytes], str, int, dict[str, Any] | None] | None:
    """Decode a stream envelope, or return None for a normal JSON output."""
    if not isinstance(payload, Mapping) or payload.get(STREAM_MARKER) is not True:
        return None
    raw_chunks = payload.get("chunks")
    if not isinstance(raw_chunks, list):
        return None
    chunks: list[bytes] = []
    try:
        for raw in raw_chunks:
            if not isinstance(raw, str):
                return None
            chunks.append(base64.b64decode(raw.encode("ascii"), validate=True))
    except (ValueError, TypeError):
        return None
    content_type = payload.get("content_type")
    status_code = payload.get("status_code")
    usage = payload.get("usage")
    return (
        chunks,
        str(content_type) if isinstance(content_type, str) else "text/event-stream",
        int(status_code) if isinstance(status_code, int) else 200,
        dict(usage) if isinstance(usage, Mapping) else None,
    )


def stream_usage(chunks: Sequence[bytes]) -> dict[str, Any] | None:
    """Collect token usage from OpenAI or Anthropic SSE chunks when present."""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    for chunk in chunks:
        for line in chunk.decode("utf-8", errors="ignore").splitlines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(event, Mapping):
                continue
            usage = event.get("usage")
            if isinstance(usage, Mapping):
                prompt_tokens = _number(usage.get("prompt_tokens"), prompt_tokens)
                completion_tokens = _number(
                    usage.get("completion_tokens"), completion_tokens
                )
                total_tokens = _number(usage.get("total_tokens"), total_tokens)
                input_tokens = _number(usage.get("input_tokens"), input_tokens)
                output_tokens = _number(usage.get("output_tokens"), output_tokens)
            message = event.get("message")
            if isinstance(message, Mapping):
                nested = message.get("usage")
                if isinstance(nested, Mapping):
                    input_tokens = _number(nested.get("input_tokens"), input_tokens)
                    output_tokens = _number(nested.get("output_tokens"), output_tokens)

    usage_out: dict[str, Any] = {}
    if prompt_tokens is not None:
        usage_out["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        usage_out["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        usage_out["total_tokens"] = total_tokens
    if input_tokens is not None:
        usage_out["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage_out["output_tokens"] = output_tokens
    if "total_tokens" not in usage_out:
        if prompt_tokens is not None or completion_tokens is not None:
            usage_out["total_tokens"] = (prompt_tokens or 0) + (completion_tokens or 0)
        elif input_tokens is not None or output_tokens is not None:
            usage_out["total_tokens"] = (input_tokens or 0) + (output_tokens or 0)
    return usage_out or None


def _number(value: Any, current: int | None) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else current
