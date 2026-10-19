"""State snapshot recording and replay for Tier 1 before/after diffs."""

from __future__ import annotations

from typing import Any

import msgspec

from checkpoint.sdk import client as sdk_client
from checkpoint.sdk import determinism
from checkpoint.sdk.determinism import real_now_iso


def checkpoint(name: str, state: Any) -> Any:
    """Record a state snapshot, or return the recorded state during replay.

    ``state`` must be JSON serializable. In replay the returned value is the
    state captured in the cassette, so agents can restore it instead of
    trusting locally recomputed state.
    """
    runtime = sdk_client.run_state()
    if runtime is None:
        return state
    state_obj = msgspec.json.decode(msgspec.json.encode(state))
    controller = determinism.controller()
    boundary = 0
    if controller is not None:
        boundary = controller.begin_boundary()
    if runtime.mode in ("REPLAY", "FORK"):
        response = sdk_client.lookup_step(
            {
                "run_id": runtime.run_id,
                "step_type": "CHECKPOINT",
                "name": name,
                "input": {},
                "boundary_index": boundary,
            }
        )
        if response.get("status") == "replayed":
            sdk_client.settle(response)
            return response.get("output")
        if response.get("status") == "replayed_error":
            sdk_client.settle(response)
            raise RuntimeError(f"recorded checkpoint error: {response.get('error')}")
    info = sdk_client.record_step(
        {
            "run_id": runtime.run_id,
            "step_type": "CHECKPOINT",
            "name": name,
            "input": {},
            "output": state_obj,
            "source": "RECORDED" if runtime.mode == "RECORD" else "LIVE",
            "frozen_timestamp": real_now_iso(),
        }
    )
    sdk_client.settle(info)
    return state
