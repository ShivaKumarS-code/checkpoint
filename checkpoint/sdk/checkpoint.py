"""State snapshot recording for Tier 1 before/after diffs."""

from __future__ import annotations

from typing import Any

import msgspec

from checkpoint.sdk import client as sdk_client


def checkpoint(name: str, state: Any) -> None:
    """Record a JSON-serializable state snapshot as a CHECKPOINT step."""
    runtime = sdk_client.run_state()
    if runtime is None or runtime.mode != "RECORD":
        return
    state_obj = msgspec.json.decode(msgspec.json.encode(state))
    sdk_client.record_step(
        {
            "run_id": runtime.run_id,
            "step_type": "CHECKPOINT",
            "name": name,
            "input": {},
            "output": state_obj,
        }
    )
