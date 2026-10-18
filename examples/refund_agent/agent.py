"""Reference refund agent: 3 LLM calls, 4 tools (one stub), 1 checkpoint.

Boundary steps, in order:

0. LLM ``plan``: read the ticket, extract the order id.
1. tool ``lookup_order`` (pure): deterministic in-memory order lookup.
2. tool ``check_policy`` (pure): refund policy check.
3. LLM ``decide``: approve or deny the refund.
4. tool ``confirm_with_user`` (pure): confirmation, driven by env for tests.
5. tool ``issue_refund`` (stub): writes the refund ledger, has a side effect.
6. LLM ``compose_reply``: final customer message.
7. checkpoint ``after_refund``: the state dict below.

The baseline decision prompt does not mention prior refunds on purpose, so
tickets for orders with a ``prior_refund_id`` reproduce a double-refund bug
that fork variants can fix (see SPEC.md section 9).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:4040/v1"
DEFAULT_MODEL = "demo-model"
DEFAULT_TICKET = "Please refund order ORD-1002. It arrived damaged."
REFUND_LIMIT = 100.0

PLANNER_SYSTEM = "You are a refund planner. Reply with JSON containing order_id and action."
DECISION_SYSTEM = (
    "You are a refund decision engine. Approve only if the policy says the order is refundable."
)
REPLY_SYSTEM = "You are a support reply writer. Write one short sentence for the customer."

ORDERS: dict[str, dict[str, Any]] = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "customer": "ada@example.com",
        "amount": 42.0,
        "status": "delivered",
        "prior_refund_id": None,
    },
    "ORD-1002": {
        "order_id": "ORD-1002",
        "customer": "grace@example.com",
        "amount": 84.5,
        "status": "delivered",
        "prior_refund_id": "rf_old_1002",
    },
    "ORD-1003": {
        "order_id": "ORD-1003",
        "customer": "linus@example.com",
        "amount": 120.0,
        "status": "shipped",
        "prior_refund_id": None,
    },
}


def lookup_order(order_id: str) -> dict[str, Any]:
    """Tool (pure): deterministic in-memory order lookup."""
    return ORDERS.get(order_id, {"error": f"unknown order_id {order_id}"})


def check_policy(order: dict[str, Any]) -> dict[str, Any]:
    """Tool (pure): refund policy check."""
    amount = order.get("amount")
    status = order.get("status")
    refundable = (
        isinstance(amount, (int, float))
        and amount <= REFUND_LIMIT
        and status in {"delivered", "shipped"}
    )
    return {"refundable": refundable, "amount": amount, "limit": REFUND_LIMIT}


def confirm_with_user(order_id: str, amount: float | None) -> bool:
    """Tool (pure): confirmation step, driven by env so CI stays non-interactive."""
    del order_id, amount
    answer = os.environ.get("REFUND_AGENT_AUTO_CONFIRM", "yes").strip().lower()
    return answer in {"1", "true", "yes"}


def issue_refund(order_id: str, amount: float) -> dict[str, Any]:
    """Tool (stub): side-effecting refund; appends to the ledger file."""
    refund_id = f"rf_{order_id.lower().replace('-', '_')}"
    entry = {"refund_id": refund_id, "order_id": order_id, "amount": amount}
    ledger = os.environ.get("REFUND_AGENT_LEDGER", "refund_ledger.jsonl")
    with open(ledger, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return {"refund_id": refund_id, "status": "refunded"}


def _chat(task: str, system: str, payload: dict[str, Any]) -> str:
    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    headers = {"Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "sk-fake-key")}
    body = {
        "model": os.environ.get("REFUND_AGENT_MODEL", DEFAULT_MODEL),
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({"task": task, **payload}, sort_keys=True)},
        ],
    }
    response = httpx.post(
        f"{base_url}/chat/completions", json=body, headers=headers, timeout=30.0
    )
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"])


def handle_ticket(ticket: str) -> dict[str, Any]:
    """Run the agent end to end and return its final state."""
    plan = json.loads(_chat("plan", PLANNER_SYSTEM, {"ticket": ticket}))
    order = lookup_order(str(plan.get("order_id")))
    policy = check_policy(order)
    decision = json.loads(_chat("decide", DECISION_SYSTEM, {"order": order, "policy": policy}))

    confirmation: bool | None = None
    refund: dict[str, Any] | None = None
    if decision.get("decision") == "approve":
        confirmation = confirm_with_user(str(order.get("order_id")), decision.get("amount"))
        amount = decision.get("amount")
        if confirmation and isinstance(amount, (int, float)):
            refund = issue_refund(str(order.get("order_id")), float(amount))

    reply = _chat(
        "compose",
        REPLY_SYSTEM,
        {
            "decision": decision.get("decision"),
            "amount": decision.get("amount"),
            "refund_id": (refund or {}).get("refund_id"),
        },
    )
    return {
        "ticket": ticket,
        "plan": plan,
        "order": order,
        "policy": policy,
        "decision": decision,
        "confirmation": confirmation,
        "refund": refund,
        "reply": reply,
    }


def main() -> int:
    ticket = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TICKET
    state = handle_ticket(ticket)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
