# checkpoint

**Deterministic replay and fork debugging for AI agents.**

`checkpoint` records every non-deterministic boundary of an AI agent run (LLM
calls, tool calls, clock, UUIDs, random seeds) into a local SQLite cassette. You
can replay a run, fork it at a step, change the prompt, model, or code from that
point on, and grade several variants against the same recorded prefix.

This is **replay fork**, not a process snapshot: the agent's local Python code
still re-executes. Boundary calls before the fork point are served from the
cassette, so they make no upstream calls and cost no tokens. See the
[full README](https://github.com/ShivaKumarS-code/checkpoint#readme) for the
honest scope, benchmarks, case study, and known limits.

## Install

The package is prepared for PyPI as `checkpoint-ai`; the v0.1.0 upload is still
pending. If the release is available:

```bash
pip install checkpoint-ai
checkpoint web
```

Until then, install from a source checkout:

```bash
git clone https://github.com/ShivaKumarS-code/checkpoint.git
cd checkpoint/backend
uv sync
uv run checkpoint web
```

The UI is at <http://127.0.0.1:4040>. The package installs the `checkpoint`
command; importing the Python SDK still uses `import checkpoint`.

## Quickstart

Start the deterministic fake provider from a source checkout:

```bash
cd backend
uv run python -m examples.fake_llm
```

Then record the demo agent in another terminal:

```bash
cd backend
uv run checkpoint run \
  --upstream http://127.0.0.1:4141 \
  --name refund-agent \
  -- python examples/refund_agent/agent.py \
  "Please refund order ORD-1002. It arrived damaged."
```

Replay the printed run id:

```bash
uv run checkpoint replay <run_id> \
  python examples/refund_agent/agent.py \
  "Please refund order ORD-1002. It arrived damaged."
```

Fork it across variants:

```bash
uv run checkpoint fork <run_id> \
  --variants examples/refund_agent/variants.yaml
```

## Checks

- Tier 1: deterministic checks over steps, state, and final output.
- Tier 2: cached rubric judge calls.
- Tier 3: swapped-order pairwise judging with a stored
  `position_consistent` flag; order-biased verdicts fail.

```yaml
checks:
  - type: pairwise
    name: quality
    rubric: "Prefer the correct, specific, safe reply."
    against: parent
    allow_tie: false
```

## Limits

Sequential agents only. Tool inputs/outputs and checkpoint state must be
JSON-serializable. Clock freezing affects timeouts and elapsed-time code.
Replay fork re-executes local code and only skips boundary calls. Semantic
matching is opt-in because it can hide regressions. The daemon binds to
`127.0.0.1` only.

## Development

```bash
cd backend
uv sync
uv run pytest -q
uv run ruff check .
uv run pyright
```

No real API keys are required by the test suite or the demo.
