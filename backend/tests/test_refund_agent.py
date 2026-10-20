"""End-to-end tests of the demo refund agent against the fake LLM."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples import fake_llm
from examples.refund_agent import agent


def prepare(
    server: fake_llm.FakeLLMServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    fake_llm.reset()
    monkeypatch.setenv("OPENAI_BASE_URL", f"{server.url}/v1")
    ledger = tmp_path / "refund_ledger.jsonl"
    monkeypatch.setenv("REFUND_AGENT_LEDGER", str(ledger))
    return ledger


def test_agent_makes_three_llm_calls_and_issues_a_refund(
    fake_llm_server: fake_llm.FakeLLMServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = prepare(fake_llm_server, tmp_path, monkeypatch)
    result = agent.handle_ticket("Please refund order ORD-1002. It arrived damaged.")

    assert fake_llm.call_count() == 3
    assert result["plan"]["order_id"] == "ORD-1002"
    assert result["decision"]["decision"] == "approve"
    assert result["confirmation"] is True
    assert result["refund"]["refund_id"] == "rf_ord_1002"
    assert "rf_ord_1002" in result["reply"]

    entries = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert entries == [
        {"amount": 84.5, "order_id": "ORD-1002", "refund_id": "rf_ord_1002"},
    ]


def test_agent_skips_refund_without_user_confirmation(
    fake_llm_server: fake_llm.FakeLLMServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = prepare(fake_llm_server, tmp_path, monkeypatch)
    monkeypatch.setenv("REFUND_AGENT_AUTO_CONFIRM", "no")
    result = agent.handle_ticket("Please refund order ORD-1001, it is late.")

    assert result["decision"]["decision"] == "approve"
    assert result["confirmation"] is False
    assert result["refund"] is None
    assert not ledger.exists()


def test_agent_denies_orders_over_the_limit(
    fake_llm_server: fake_llm.FakeLLMServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = prepare(fake_llm_server, tmp_path, monkeypatch)
    result = agent.handle_ticket("Please refund order ORD-1003, it has not arrived.")

    assert result["policy"]["refundable"] is False
    assert result["decision"]["decision"] == "deny"
    assert result["refund"] is None
    assert not ledger.exists()
