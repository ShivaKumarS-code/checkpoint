"""Reproducible live-vs-replay benchmark for checkpoint.

The benchmark uses the deterministic fake provider (no API keys) and a
controlled per-call latency so replay savings are measurable even on localhost.
It measures end-to-end agent subprocess wall time and upstream token usage for:

* ``demo``: the three-call refund agent in ``examples/refund_agent``.
* ``diffguardian``: the real DiffGuardian LangGraph workflow, using the
  optional adapter in ``benchmarks/diffguardian_target.py``.

DiffGuardian is optional. Point ``DIFFGUARDIAN_ROOT`` at the checkout and
``DIFFGUARDIAN_PYTHON`` at an interpreter with DiffGuardian's dependencies plus
checkpoint installed. The default demo target needs nothing beyond checkpoint's
own environment.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from checkpoint.daemon.app import run_daemon
from checkpoint.daemon.recorder import Recorder
from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import init_db
from examples import fake_llm

BACKEND_ROOT = Path(__file__).resolve().parents[1]
HARNESS = BACKEND_ROOT / "benchmarks" / "diffguardian_target.py"
DEMO_TARGET = BACKEND_ROOT / "benchmarks" / "demo_target.py"
TICKET = "Please refund order ORD-1002. It arrived damaged."
FAKE_DELAY_MS = 1000.0


@dataclass
class Target:
    name: str
    python: str
    command: list[str]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)


def _clean_env(extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("CHECKPOINT_"):
            env.pop(key)
    env.update(
        {
            "OPENAI_API_KEY": "sk-fake-benchmark",
            "ANTHROPIC_API_KEY": "sk-ant-fake-benchmark",
            "GROQ_API_KEY": "fake-groq-benchmark",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(BACKEND_ROOT),
        }
    )
    env.update(extra)
    return env


def _spawn(
    target: Target,
    daemon_url: str,
    run_id: str,
    mode: str,
    parent_run_id: str | None,
    ledger: Path,
    bench_file: Path,
) -> tuple[int, float, str, str]:
    env = _clean_env(target.env)
    env.update(
        {
            "CHECKPOINT_DAEMON_URL": daemon_url,
            "CHECKPOINT_RUN_ID": run_id,
            "CHECKPOINT_MODE": mode,
            "CHECKPOINT_BENCHMARK_FILE": str(bench_file),
            "OPENAI_BASE_URL": f"{daemon_url}/v1",
            "ANTHROPIC_BASE_URL": f"{daemon_url}/v1",
        }
    )
    if parent_run_id is not None:
        env["CHECKPOINT_PARENT_RUN_ID"] = parent_run_id
    if target.name == "demo":
        env["REFUND_AGENT_LEDGER"] = str(ledger)
    started = time.perf_counter()
    completed = subprocess.run(
        target.command,
        cwd=target.cwd,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=300,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return completed.returncode, elapsed_ms, completed.stdout, completed.stderr


def _create_run(
    db_path: Path, name: str, mode: str, parent: Any | None = None
) -> str:
    conn = init_db(db_path)
    try:
        recorder = Recorder(conn, BlobStore(conn))
        created = recorder.create_run(
            name,
            mode,
            parent_run_id=str(parent["run_id"]) if parent is not None else None,
            run_seed=int(parent["run_seed"]) if parent is not None else None,
        )
        return str(created["run_id"])
    finally:
        conn.close()


def _finish_run(db_path: Path, run_id: str, returncode: int) -> dict[str, Any]:
    conn = init_db(db_path)
    try:
        recorder = Recorder(conn, BlobStore(conn))
        recorder.finish_run(run_id, "COMPLETED" if returncode == 0 else "FAILED")
        run = recorder.get_run(run_id)
        steps = recorder.steps_for(run_id)
        return {
            "status": str(run["status"]) if run is not None else "FAILED",
            "tokens": int(run["total_tokens"]) if run is not None else 0,
            "latency_ms": float(run["total_latency_ms"]) if run is not None else 0.0,
            "upstream_calls": sum(
                1
                for step in steps
                if step["step_type"] == "LLM_CALL"
                and step["source"] in ("RECORDED", "LIVE")
            ),
            "steps": len(steps),
        }
    finally:
        conn.close()


def _measure(
    db_path: Path,
    daemon_url: str,
    target: Target,
    mode: str,
    parent: Any | None,
    ledger: Path,
) -> dict[str, Any]:
    run_id = _create_run(db_path, target.name, mode, parent)
    bench_file = ledger.parent / f"bench-{run_id}.json"
    bench_file.unlink(missing_ok=True)
    fake_llm.reset()
    returncode, wall_ms, stdout, stderr = _spawn(
        target,
        daemon_url,
        run_id,
        "record" if mode == "RECORD" else "replay",
        str(parent["run_id"]) if parent is not None else None,
        ledger,
        bench_file,
    )
    info = _finish_run(db_path, run_id, returncode)
    agent_ms: float | None = None
    if bench_file.is_file():
        try:
            payload = json.loads(bench_file.read_text(encoding="utf-8"))
            value = payload.get("agent_ms")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                agent_ms = float(value)
        except (OSError, ValueError):
            agent_ms = None
    info.update(
        {
            "run_id": run_id,
            "returncode": returncode,
            "wall_ms": wall_ms,
            "agent_ms": agent_ms,
            "provider_calls": fake_llm.call_count(),
            "stdout_tail": stdout[-500:],
            "stderr_tail": stderr[-1000:],
        }
    )
    return info


def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    walls = sorted(float(sample["wall_ms"]) for sample in samples)
    agent_times = sorted(
        float(sample["agent_ms"])
        for sample in samples
        if sample.get("agent_ms") is not None
    )
    tokens = sorted(int(sample["tokens"]) for sample in samples)
    return {
        "runs": len(samples),
        "agent_ms_median": (
            round(statistics.median(agent_times), 2) if agent_times else None
        ),
        "agent_ms_min": round(min(agent_times), 2) if agent_times else None,
        "agent_ms_max": round(max(agent_times), 2) if agent_times else None,
        "wall_ms_median": round(statistics.median(walls), 2),
        "wall_ms_min": round(min(walls), 2),
        "wall_ms_max": round(max(walls), 2),
        "tokens_median": int(statistics.median(tokens)),
        "tokens_min": min(tokens),
        "tokens_max": max(tokens),
        "upstream_calls": sum(int(sample["upstream_calls"]) for sample in samples),
        "samples": samples,
    }


def _demo_target() -> Target:
    return Target(
        name="demo",
        python=sys.executable,
        command=[sys.executable, str(DEMO_TARGET)],
        cwd=BACKEND_ROOT,
        env={"CHECKPOINT_BENCH_TICKET": TICKET},
    )


def _diffguardian_target(root: Path, python: str) -> Target:
    if not (root / "app" / "services" / "graph.py").is_file():
        raise SystemExit(f"not a DiffGuardian checkout: {root}")
    probe = subprocess.run(
        [python, "-c", "import langgraph, langchain_groq, groq, checkpoint"],
        cwd=BACKEND_ROOT,
        env=_clean_env({"PYTHONPATH": str(BACKEND_ROOT)}),
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise SystemExit(
            "DiffGuardian target is unavailable. Install DiffGuardian's "
            "dependencies and checkpoint into DIFFGUARDIAN_PYTHON.\n"
            + probe.stderr[-1500:]
        )
    return Target(
        name="diffguardian",
        python=python,
        command=[python, str(HARNESS)],
        cwd=BACKEND_ROOT,
        env={
            "DIFFGUARDIAN_ROOT": str(root),
            "CHECKPOINT_BENCHMARK_TARGET": "diffguardian",
        },
    )


def _run_target(
    target: Target, runs: int, db_path: Path, daemon_url: str, work_dir: Path
) -> dict[str, Any]:
    ledger = work_dir / f"{target.name}-live-ledger.jsonl"
    warmup = _measure(
        db_path, daemon_url, target, "RECORD", None, ledger
    )
    parent_conn = init_db(db_path)
    try:
        parent = Recorder(parent_conn, BlobStore(parent_conn)).get_run(
            warmup["run_id"]
        )
    finally:
        parent_conn.close()
    if parent is None:
        raise RuntimeError("warmup run disappeared")
    replay_warmup = _measure(db_path, daemon_url, target, "REPLAY", parent, ledger)
    if replay_warmup["provider_calls"] != 0:
        raise RuntimeError("replay contacted the fake provider")

    live_samples = [
        _measure(db_path, daemon_url, target, "RECORD", None, ledger)
        for _ in range(runs)
    ]
    replay_samples = [
        _measure(db_path, daemon_url, target, "REPLAY", parent, ledger)
        for _ in range(runs)
    ]
    if any(sample["provider_calls"] != 0 for sample in replay_samples):
        raise RuntimeError("a replay measurement contacted the fake provider")
    if any(sample["status"] != "COMPLETED" for sample in [*live_samples, *replay_samples]):
        raise RuntimeError("a measured run did not complete")
    return {
        "live": _summary(live_samples),
        "replay": _summary(replay_samples),
        "warmup": {"live": warmup, "replay": replay_warmup},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=["demo", "diffguardian", "all"],
        default="demo",
        help="which agent target to measure",
    )
    parser.add_argument("--runs", type=int, default=5, help="measured runs per mode")
    parser.add_argument(
        "--delay-ms",
        type=float,
        default=FAKE_DELAY_MS,
        help="simulated per-call provider latency in milliseconds",
    )
    parser.add_argument(
        "--diffguardian-root",
        type=Path,
        default=Path(os.environ.get("DIFFGUARDIAN_ROOT", "D:/diffguardian")),
    )
    parser.add_argument(
        "--diffguardian-python",
        default=os.environ.get("DIFFGUARDIAN_PYTHON", sys.executable),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=BACKEND_ROOT / "benchmarks" / "results.json",
    )
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")

    os.environ["FAKE_LLM_DELAY_MS"] = str(max(0.0, args.delay_ms))
    fake = fake_llm.serve()
    results: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": {
            "provider": "deterministic fake OpenAI-compatible server",
            "simulated_provider_delay_ms_per_call": args.delay_ms,
            "measured_runs_per_mode": args.runs,
            "agent_time": "wall time around agent execution, excluding interpreter startup",
            "process_wall_time": "subprocess wall time, including interpreter startup",
            "tokens": "run total_tokens; replayed and semantic steps count as zero",
            "warmup": "one record + replay discarded before measurement",
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "checkpoint_python": sys.executable,
        },
        "targets": {},
    }
    try:
        with tempfile.TemporaryDirectory(prefix="checkpoint-bench-") as raw_dir:
            work_dir = Path(raw_dir)
            db_path = work_dir / "cassettes.db"
            daemon = run_daemon(db_path, fake.url, port=0)
            try:
                targets: list[Target] = []
                if args.target in ("demo", "all"):
                    targets.append(_demo_target())
                if args.target in ("diffguardian", "all"):
                    targets.append(
                        _diffguardian_target(
                            args.diffguardian_root, args.diffguardian_python
                        )
                    )
                for target in targets:
                    print(f"benchmarking {target.name}...", file=sys.stderr)
                    results["targets"][target.name] = _run_target(
                        target, args.runs, db_path, daemon.url, work_dir
                    )
            finally:
                daemon.stop()
    finally:
        fake.stop()
        os.environ.pop("FAKE_LLM_DELAY_MS", None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
