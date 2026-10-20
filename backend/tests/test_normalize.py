"""Tests for request normalization and match_key hashing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import pytest

from checkpoint.config import DEFAULT_CONFIG_FILENAME, effective_masks, load_config
from checkpoint.matching.normalize import (
    DEFAULT_MASKS,
    canonical_json,
    match_key,
    normalize_request,
)


def make_body(content: str) -> dict[str, Any]:
    return {
        "model": "demo-model",
        "temperature": 0.0,
        "messages": [{"role": "user", "content": content}],
    }


def test_canonical_json_sorts_keys_and_stays_compact() -> None:
    assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_canonical_json_keeps_non_ascii_as_utf8() -> None:
    assert canonical_json({"x": "é"}) == '{"x":"é"}'.encode()


def test_requests_differing_only_in_masked_timestamp_have_same_match_key() -> None:
    first = make_body("Order placed at 2026-10-18T10:00:00Z")
    second = make_body("Order placed at 2027-01-02T23:59:59.123456+05:30")
    assert match_key(first) == match_key(second)


def test_requests_differing_only_in_masked_uuid_have_same_match_key() -> None:
    first = make_body("session 123e4567-e89b-12d3-a456-426614174000")
    second = make_body("session 00000000-0000-4000-8000-000000000000")
    assert match_key(first) == match_key(second)


def test_different_content_gives_different_match_key() -> None:
    assert match_key(make_body("refund approved")) != match_key(make_body("refund denied"))


def test_match_key_is_sha256_hex() -> None:
    key = match_key(make_body("x"))
    assert len(key) == 64
    int(key, 16)


def test_volatile_top_level_fields_are_dropped() -> None:
    base = make_body("hello")
    noisy = {
        **base,
        "user": "user-42",
        "stream_options": {"include_usage": True},
        "request_id": "req_1",
    }
    assert normalize_request(noisy) == normalize_request(base)
    assert match_key(noisy) == match_key(base)


def test_volatile_key_names_are_kept_inside_message_payloads() -> None:
    body: dict[str, Any] = {"messages": [{"role": "user", "content": "user"}]}
    normalized = normalize_request(body)
    assert normalized["messages"][0]["role"] == "user"


def test_masks_apply_to_nested_strings() -> None:
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": {"note": "at 2026-10-18T10:00:00Z"}}]
    }
    normalized = normalize_request(body)
    assert normalized["messages"][0]["content"]["note"] == "at <timestamp>"


def test_non_string_values_are_unchanged() -> None:
    body: dict[str, Any] = {"seed": 42, "flag": True, "nothing": None, "count": 5.5}
    assert normalize_request(body) == body


def test_config_masks_extend_builtin_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / DEFAULT_CONFIG_FILENAME
    config_file.write_text(
        '[[masks]]\npattern = "sess_[0-9a-z]{8}"\nreplacement = "<session>"\n',
        encoding="utf-8",
    )
    config = load_config(config_file)
    masks = effective_masks(config)
    assert masks[: len(DEFAULT_MASKS)] == DEFAULT_MASKS
    assert len(masks) == len(DEFAULT_MASKS) + 1

    body: dict[str, Any] = {"content": "session id sess_ab12cd34"}
    assert normalize_request(body, masks) == {"content": "session id <session>"}


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config_file = tmp_path / DEFAULT_CONFIG_FILENAME
    config_file.write_text("bogus = 1\n", encoding="utf-8")
    with pytest.raises(msgspec.ValidationError):
        load_config(config_file)


def test_missing_default_config_returns_empty_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert config.masks == []
    assert effective_masks(config) == DEFAULT_MASKS


def test_repo_example_config_loads_with_extra_masks() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / DEFAULT_CONFIG_FILENAME)
    assert len(config.masks) == 1
    assert config.masks[0].replacement == "<session>"
    masks = effective_masks(config)
    assert masks[: len(DEFAULT_MASKS)] == DEFAULT_MASKS
