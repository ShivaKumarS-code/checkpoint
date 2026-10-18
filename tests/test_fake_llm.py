"""Tests for the deterministic fake LLM."""

from __future__ import annotations

import json
from typing import Any

import httpx

from examples import fake_llm


def post_chat(
    server: fake_llm.FakeLLMServer,
    messages: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    body = {"model": "demo-model", "messages": messages, **extra}
    response = httpx.post(f"{server.url}/v1/chat/completions", json=body, timeout=10.0)
    response.raise_for_status()
    return response.json()


def completion_text(response: dict[str, Any]) -> str:
    return response["choices"][0]["message"]["content"]


def test_identical_requests_get_identical_responses(
    fake_llm_server: fake_llm.FakeLLMServer,
) -> None:
    messages = [
        {"role": "system", "content": "planner"},
        {"role": "user", "content": json.dumps({"task": "plan", "ticket": "refund ORD-1002"})},
    ]
    first = post_chat(fake_llm_server, messages)
    second = post_chat(fake_llm_server, messages)
    assert first == second


def test_generate_completion_is_a_pure_function() -> None:
    body = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": json.dumps({"task": "plan", "ticket": "ORD-7"})}],
    }
    assert fake_llm.generate_completion(body) == fake_llm.generate_completion(body)


def test_stats_count_and_reset(fake_llm_server: fake_llm.FakeLLMServer) -> None:
    fake_llm.reset()
    assert fake_llm.call_count() == 0
    post_chat(fake_llm_server, [{"role": "user", "content": "hi"}])
    assert fake_llm.call_count() == 1
    stats = httpx.get(f"{fake_llm_server.url}/__control/stats", timeout=10.0).json()
    assert stats == {"calls": 1}
    reset = httpx.post(f"{fake_llm_server.url}/__control/reset", timeout=10.0).json()
    assert reset == {"calls": 0}


def test_plan_task_extracts_order_id(fake_llm_server: fake_llm.FakeLLMServer) -> None:
    user_message = json.dumps({"task": "plan", "ticket": "please refund ORD-1234"})
    response = post_chat(fake_llm_server, [{"role": "user", "content": user_message}])
    assert json.loads(completion_text(response)) == {"order_id": "ORD-1234", "action": "refund"}


def test_decide_respects_strict_system_prompt(fake_llm_server: fake_llm.FakeLLMServer) -> None:
    user = json.dumps(
        {
            "task": "decide",
            "order": {"prior_refund_id": "rf_old", "status": "delivered"},
            "policy": {"refundable": True, "amount": 10.0},
        }
    )
    baseline = [
        {"role": "system", "content": "Approve if the policy says refundable."},
        {"role": "user", "content": user},
    ]
    strict = [
        {"role": "system", "content": "Never approve when prior_refund_id is set."},
        {"role": "user", "content": user},
    ]
    baseline_decision = json.loads(completion_text(post_chat(fake_llm_server, baseline)))
    strict_decision = json.loads(completion_text(post_chat(fake_llm_server, strict)))
    assert baseline_decision["decision"] == "approve"
    assert strict_decision["decision"] == "deny"


def test_compose_mentions_refund_id(fake_llm_server: fake_llm.FakeLLMServer) -> None:
    response = post_chat(
        fake_llm_server,
        [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "compose",
                        "decision": "approve",
                        "amount": 84.5,
                        "refund_id": "rf_ord_1002",
                    }
                ),
            }
        ],
    )
    assert completion_text(response) == "Refund rf_ord_1002 of 84.50 has been issued."


def test_invalid_body_returns_400(fake_llm_server: fake_llm.FakeLLMServer) -> None:
    response = httpx.post(
        f"{fake_llm_server.url}/v1/chat/completions", json={"model": "x"}, timeout=10.0
    )
    assert response.status_code == 400
