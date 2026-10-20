"""Canonicalization and masking of boundary requests for match_key hashing.

Headers are never part of a match_key: normalization only ever sees the
request body. Known volatile top-level fields are dropped, then regex masks
are applied to every string value in the body.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

import msgspec


class MaskRule(msgspec.Struct, frozen=True):
    """A regex applied to every string value of a request body."""

    pattern: str
    replacement: str = "<masked>"


DEFAULT_MASKS: tuple[MaskRule, ...] = (
    MaskRule(
        pattern=r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?",
        replacement="<timestamp>",
    ),
    MaskRule(
        pattern=r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
        replacement="<uuid>",
    ),
)

VOLATILE_TOP_LEVEL_KEYS = frozenset({"user", "stream_options", "request_id"})


def canonical_json(value: Any) -> bytes:
    """Canonical JSON: sorted keys, compact separators, UTF-8 bytes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _mask_value(value: Any, masks: tuple[MaskRule, ...]) -> Any:
    if isinstance(value, str):
        for mask in masks:
            value = re.sub(mask.pattern, mask.replacement, value)
        return value
    if isinstance(value, Mapping):
        return {key: _mask_value(item, masks) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mask_value(item, masks) for item in value]
    return value


def normalize_request(
    body: Mapping[str, Any], masks: Iterable[MaskRule] | None = None
) -> dict[str, Any]:
    """Normalize a request body: drop volatile fields, apply masks.

    ``masks=None`` uses the built-in timestamp and UUID masks.
    """
    active = tuple(masks) if masks is not None else DEFAULT_MASKS
    return {
        key: _mask_value(value, active)
        for key, value in body.items()
        if key not in VOLATILE_TOP_LEVEL_KEYS
    }


def match_key(body: Mapping[str, Any], masks: Iterable[MaskRule] | None = None) -> str:
    """SHA-256 hex digest of the canonical normalized request body."""
    return hashlib.sha256(canonical_json(normalize_request(body, masks))).hexdigest()
