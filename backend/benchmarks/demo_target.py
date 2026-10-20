"""Benchmark wrapper for the refund demo.

Importing the agent installs checkpoint determinism and the proxy patch, then the
timer covers only ``handle_ticket`` so interpreter startup does not dominate the
measurement.
"""

from __future__ import annotations

import json
import os
import time

from examples.refund_agent.agent import DEFAULT_TICKET, handle_ticket


def main() -> int:
    ticket = os.environ.get("CHECKPOINT_BENCH_TICKET", DEFAULT_TICKET)
    started = time.perf_counter()
    state = handle_ticket(ticket)
    agent_ms = (time.perf_counter() - started) * 1000.0
    output = os.environ.get("CHECKPOINT_BENCHMARK_FILE")
    if output:
        with open(output, "w", encoding="utf-8") as handle:
            json.dump({"agent_ms": agent_ms, "reply": state.get("reply")}, handle)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
