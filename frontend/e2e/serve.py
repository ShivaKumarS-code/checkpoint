"""Seed a deterministic cassette and serve the built UI for Playwright."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if os.environ.get("CHECKPOINT_E2E_USE_INSTALLED") != "1":
    sys.path.insert(0, str(ROOT / "backend"))

from checkpoint.daemon.app import run_daemon  # noqa: E402
from checkpoint.daemon.recorder import Recorder  # noqa: E402
from checkpoint.storage.cas import BlobStore  # noqa: E402
from checkpoint.storage.db import init_db  # noqa: E402

PROMPT = (
    "Please refund order ORD-1002. It arrived damaged. "
    "<script>window.__xss=1</script>"
)
PLAN_REQUEST = {
    "model": "demo-model",
    "temperature": 0.0,
    "messages": [
        {"role": "system", "content": "You are a refund planner. Reply with JSON."},
        {"role": "user", "content": PROMPT},
    ],
}
PLAN_RESPONSE = {
    "id": "chatcmpl-e2e-plan",
    "object": "chat.completion",
    "created": 1,
    "model": "demo-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": '{"order_id": "ORD-1002", "action": "refund"}',
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}
LOOKUP_INPUT = {"args": ["ORD-1002"], "kwargs": {}}
LOOKUP_OUTPUT = {
    "order_id": "ORD-1002",
    "customer": "grace@example.com",
    "amount": 84.5,
    "status": "delivered",
    "prior_refund_id": "rf_old_1002",
}
DECIDE_REQUEST = {
    "model": "demo-model",
    "temperature": 0.0,
    "messages": [
        {
            "role": "system",
            "content": "You are a refund decision engine. Approve only if refundable.",
        },
        {"role": "user", "content": '{"order": "ORD-1002", "amount": 84.5}'},
    ],
}
DECIDE_RESPONSE = {
    "id": "chatcmpl-e2e-decide",
    "object": "chat.completion",
    "created": 2,
    "model": "demo-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": '{"decision": "approve", "amount": 84.5}',
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
}
COMPOSE_REQUEST = {
    "model": "demo-model",
    "temperature": 0.0,
    "messages": [
        {
            "role": "system",
            "content": "You are a support reply writer. Write one short sentence.",
        },
        {"role": "user", "content": '{"decision": "deny", "refund_id": null}'},
    ],
}
COMPOSE_RESPONSE = {
    "id": "chatcmpl-e2e-compose",
    "object": "chat.completion",
    "created": 3,
    "model": "demo-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "A prior refund already exists, so I cannot issue another one.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 8, "total_tokens": 17},
}


def seed(db_path: Path) -> tuple[str, str]:
    conn = init_db(db_path)
    recorder = Recorder(conn, BlobStore(conn))
    parent = recorder.create_run("refund-agent", "RECORD")
    parent_id = str(parent["run_id"])
    recorder.record_llm_step(
        parent_id, "plan", PLAN_REQUEST, PLAN_RESPONSE, latency_ms=18.0
    )
    recorder.record_tool_step(
        parent_id,
        "lookup_order",
        LOOKUP_INPUT,
        LOOKUP_OUTPUT,
        latency_ms=0.4,
    )
    recorder.record_llm_step(
        parent_id, "decide", DECIDE_REQUEST, DECIDE_RESPONSE, latency_ms=24.0
    )
    recorder.record_checkpoint(
        parent_id,
        "after_refund",
        {"refund": {"refund_id": "rf_ord_1002", "status": "refunded"}},
    )
    recorder.finish_run(parent_id, "COMPLETED")

    experiment_id = recorder.create_experiment(
        "fork:refund-agent",
        parent_id,
        2,
        [{"name": "baseline"}, {"name": "guard_prior_refund"}],
        [{"type": "no_errors"}, {"type": "tool_not_called", "name": "issue_refund"}],
    )

    for variant, passed, reply in (
        ("baseline", False, "Refund issued."),
        ("guard_prior_refund", True, "A prior refund already exists."),
    ):
        child = recorder.create_run(
            f"refund-agent / {variant}",
            "FORK",
            parent_run_id=parent_id,
            forked_at_step=2,
            variant_name=variant,
            experiment_id=experiment_id,
        )
        child_id = str(child["run_id"])
        recorder.record_llm_step(
            child_id,
            "plan",
            PLAN_REQUEST,
            PLAN_RESPONSE,
            latency_ms=1.0,
            source="REPLAYED",
        )
        recorder.record_tool_step(
            child_id,
            "lookup_order",
            LOOKUP_INPUT,
            LOOKUP_OUTPUT,
            latency_ms=0.2,
            source="REPLAYED",
        )
        recorder.record_llm_step(
            child_id,
            "compose",
            COMPOSE_REQUEST,
            {
                **COMPOSE_RESPONSE,
                "id": f"chatcmpl-e2e-{variant}",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
            },
            latency_ms=3.0,
            source="LIVE",
        )
        recorder.record_checkpoint(
            child_id,
            "after_refund",
            {"refund": None if passed else {"refund_id": "rf_ord_1002"}},
            source="LIVE",
        )
        recorder.finish_run(child_id, "COMPLETED")
        recorder.record_verdict(
            child_id,
            "no_errors",
            passed,
            {"detail": "clean compose step" if passed else "refund issued twice"},
            tier="TIER1",
        )
        recorder.record_verdict(
            child_id,
            "tool_not_called:issue_refund",
            passed,
            {"detail": "stub was not called" if passed else "stub was called"},
            tier="TIER1",
        )
        recorder.record_verdict(
            child_id,
            "pairwise:quality",
            passed,
            {
                "winner": "a" if passed else "b",
                "position_consistent": passed,
                "orders": [
                    {"order": "a_vs_b", "mapped_winner": "a" if passed else "b"},
                    {"order": "b_vs_a", "mapped_winner": "a" if passed else None},
                ],
            },
            tier="TIER3",
            score=1.0 if passed else None,
            judge_model="fake-judge",
            compared_run_id=parent_id,
            position_consistent=passed,
        )

    conn.close()
    return parent_id, experiment_id


def main() -> None:
    port = int(os.environ.get("CHECKPOINT_E2E_PORT", "4173"))
    data_dir = Path(tempfile.mkdtemp(prefix="checkpoint-e2e-"))
    db_path = data_dir / "cassettes.db"
    parent_id, experiment_id = seed(db_path)
    print(f"seeded parent={parent_id} experiment={experiment_id}", flush=True)
    daemon = run_daemon(db_path, "http://127.0.0.1:1", port=port)
    print(f"checkpoint e2e daemon at {daemon.url}", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        daemon.stop()


if __name__ == "__main__":
    main()
