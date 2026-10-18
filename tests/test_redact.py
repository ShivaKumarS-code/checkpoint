"""Tests for secret redaction."""

from __future__ import annotations

from typing import Any

from checkpoint.matching.redact import RedactionRule, redact_text, redact_value

TEST_KEY = "sk-test-0123456789abcdef0123"
ANTHROPIC_KEY = "sk-ant-api03-abcdefghijklmnop"


def test_openai_style_keys_are_redacted() -> None:
    redacted = redact_text(f"my key is {TEST_KEY} ok")
    assert TEST_KEY not in redacted
    assert "<redacted>" in redacted


def test_anthropic_style_keys_are_redacted() -> None:
    assert ANTHROPIC_KEY not in redact_text(f"use {ANTHROPIC_KEY}")


def test_bearer_tokens_are_redacted() -> None:
    assert "zaCELgL0imWjg" not in redact_text("Authorization: Bearer zaCELgL0imWjg==")


def test_aws_access_keys_are_redacted() -> None:
    assert "AKIAIOSFODNN7EXAMPLE" not in redact_text("aws AKIAIOSFODNN7EXAMPLE here")


def test_key_value_assignments_are_redacted() -> None:
    redacted = redact_text('{"api_key": "supersecretvalue"}')
    assert "supersecretvalue" not in redacted


def test_plain_text_is_untouched() -> None:
    text = "Please refund order ORD-1002. It arrived damaged."
    assert redact_text(text) == text


def test_redaction_applies_through_nested_structures() -> None:
    value: Any = {
        "messages": [{"role": "user", "content": f"token {TEST_KEY}"}],
        "metadata": {"nested": [TEST_KEY]},
    }
    redacted = redact_value(value)
    assert redacted["messages"][0]["content"] == "token <redacted>"
    assert redacted["metadata"]["nested"] == ["<redacted>"]


def test_custom_rules_are_applied() -> None:
    rules = (RedactionRule(pattern=r"my-secret-\d+"),)
    assert redact_text("code my-secret-42", rules) == "code <redacted>"
    assert redact_text("code my-secret-42") != "code <redacted>"
