"""Redaction of secrets before anything reaches storage or logs.

Request and response headers are never stored at all; this module guards the
body content path. Rules are regex substitutions applied to every string
value.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

import msgspec


class RedactionRule(msgspec.Struct, frozen=True):
    pattern: str
    replacement: str = "<redacted>"


DEFAULT_REDACTION: tuple[RedactionRule, ...] = (
    RedactionRule(pattern=r"\bsk-[A-Za-z0-9_-]{6,}"),
    RedactionRule(pattern=r"\bAKIA[0-9A-Z]{16}\b"),
    RedactionRule(pattern=r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]{6,}=*"),
    RedactionRule(
        pattern=(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|secret)\b[\"']?\s*[:=]\s*"
            r"[\"']?[A-Za-z0-9\-._~]{6,}"
        )
    ),
)


def redact_text(text: str, rules: Iterable[RedactionRule] | None = None) -> str:
    active = tuple(rules) if rules is not None else DEFAULT_REDACTION
    for rule in active:
        text = re.sub(rule.pattern, rule.replacement, text)
    return text


def _redact(value: Any, rules: tuple[RedactionRule, ...]) -> Any:
    if isinstance(value, str):
        return redact_text(value, rules)
    if isinstance(value, Mapping):
        return {key: _redact(item, rules) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, rules) for item in value]
    return value


def redact_value(value: Any, rules: Iterable[RedactionRule] | None = None) -> Any:
    active = tuple(rules) if rules is not None else DEFAULT_REDACTION
    return _redact(value, active)
