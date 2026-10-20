"""Opt-in semantic matching for recorded LLM requests.

This module deliberately has no external embedding dependency. It uses a
deterministic hashed bag-of-tokens embedding so it works offline, in tests, and
inside a local-first debugger. Candidate requests are scored with cosine
similarity; the caller may additionally ask a small model to confirm a hit.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from checkpoint.matching.normalize import MaskRule, canonical_json, normalize_request
from checkpoint.storage.cas import BlobStore

EMBEDDING_DIMENSIONS = 512
_WORD_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class SemanticCandidate:
    """A recorded step whose request is close to the incoming request."""

    step: sqlite3.Row
    similarity: float
    recorded_input: Any


def request_text(body: Mapping[str, Any], masks: Iterable[MaskRule] | None = None) -> str:
    """Stable text representation used for the local embedding."""
    return canonical_json(normalize_request(body, masks)).decode("utf-8", errors="replace")


def embed_text(text: str) -> tuple[float, ...]:
    """Deterministic hashed embedding with word, bigram and character features."""
    lowered = text.lower()
    words = _WORD_RE.findall(lowered)
    features: list[str] = list(words)
    features.extend(
        f"{left}::{right}" for left, right in zip(words, words[1:], strict=False)
    )
    for word in words:
        if len(word) >= 3:
            features.extend(f"#{word[index:index + 3]}" for index in range(len(word) - 2))

    vector = [0.0] * EMBEDDING_DIMENSIONS
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % EMBEDDING_DIMENSIONS
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def similarity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    masks: Iterable[MaskRule] | None = None,
) -> float:
    return cosine_similarity(
        embed_text(request_text(left, masks)), embed_text(request_text(right, masks))
    )


def find_semantic_candidate(
    conn: sqlite3.Connection,
    blobs: BlobStore,
    parent_run_id: str,
    name: str,
    incoming_input: Mapping[str, Any],
    masks: tuple[MaskRule, ...] | None = None,
    threshold: float = 0.92,
    *,
    max_step_index: int | None = None,
    step_type: str = "LLM_CALL",
) -> SemanticCandidate | None:
    """Return the closest recorded request at or above ``threshold``.

    Candidates are restricted to the same step type and name. Failed recorded
    calls are ignored because serving them would only reproduce an error.
    """
    sql = (
        "SELECT * FROM steps WHERE run_id = ? AND step_type = ? AND name = ? "
        "AND error IS NULL AND input_hash IS NOT NULL AND output_hash IS NOT NULL"
    )
    params: list[Any] = [parent_run_id, step_type, name]
    if max_step_index is not None:
        sql += " AND step_index < ?"
        params.append(max_step_index)
    sql += " ORDER BY step_index"
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return None

    incoming_stream = bool(incoming_input.get("stream"))
    incoming = embed_text(request_text(incoming_input, masks))
    best: SemanticCandidate | None = None
    for row in rows:
        try:
            recorded = blobs.get_json(str(row["input_hash"]))
        except KeyError:
            continue
        if not isinstance(recorded, Mapping):
            continue
        if bool(recorded.get("stream")) != incoming_stream:
            continue
        score = cosine_similarity(incoming, embed_text(request_text(recorded, masks)))
        if best is None or score > best.similarity:
            best = SemanticCandidate(step=row, similarity=score, recorded_input=dict(recorded))
    if best is None or best.similarity < threshold:
        return None
    return best


def judge_prompt(incoming: Mapping[str, Any], recorded: Mapping[str, Any]) -> str:
    """Prompt used by the optional semantic judge."""
    return (
        "You are validating a semantic replay match. Decide whether these two "
        "LLM requests ask for the same thing. Ignore harmless wording, ordering "
        "or formatting differences, but reject changed facts, identifiers, "
        "constraints or requested actions.\n\n"
        f"RECORDED REQUEST:\n{canonical_json(recorded).decode('utf-8')}\n\n"
        f"NEW REQUEST:\n{canonical_json(incoming).decode('utf-8')}\n\n"
        'Reply with one JSON object: {"equivalent": true|false, "reason": "..."}'
    )


def parse_judge_response(raw: str) -> tuple[bool, str]:
    """Parse a judge response, defaulting closed on anything ambiguous."""
    import json

    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        payload = json.loads(text)
    except ValueError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return False, "judge returned no JSON verdict"
        try:
            payload = json.loads(text[start : end + 1])
        except ValueError:
            return False, "judge returned invalid JSON"
    if not isinstance(payload, Mapping):
        return False, "judge verdict was not an object"
    value = payload.get("equivalent")
    if isinstance(value, bool):
        return value, str(payload.get("reason") or "")
    if isinstance(value, str) and value.lower() in {"yes", "true", "pass"}:
        return True, str(payload.get("reason") or "")
    return False, str(payload.get("reason") or "judge rejected the match")
