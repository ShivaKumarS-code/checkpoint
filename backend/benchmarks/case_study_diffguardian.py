"""Reproduce the DiffGuardian security-review case study.

The real DiffGuardian LangGraph workflow is recorded with a deterministic
provider against a fixture diff containing SQL injection. The baseline variant
misses the issue; a fork that replaces the reviewer system prompt with an
explicit security review catches it. The first LLM call is replayed from the
parent cassette; only the post-fork review call reaches the provider.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from benchmark import (
    BACKEND_ROOT,
    _create_run,
    _diffguardian_target,
    _finish_run,
    _spawn,
)

from checkpoint.daemon.app import run_daemon
from checkpoint.daemon.experiments import (
    ExperimentSpec,
    LlmOverride,
    Variant,
    run_experiment,
)
from checkpoint.daemon.recorder import Recorder
from checkpoint.evaluators.deterministic import CheckSpec
from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import init_db
from examples import fake_llm

SECURITY_PROMPT = """You are diffguardian, an autonomous AI code reviewer.
SECURITY_REVIEW: inspect the diff for SQL injection, authorization bypass,
hard-coded secrets, and unsafe deserialization.
Return ONLY the required JSON object with "summary" and "inline_comments"."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=BACKEND_ROOT / "benchmarks" / "case-study.json",
    )
    args = parser.parse_args()

    target = _diffguardian_target(args.diffguardian_root, args.diffguardian_python)
    fake = fake_llm.serve()
    results: dict[str, Any] = {
        "agent": "DiffGuardian",
        "fixture": "SQL injection in app/auth.py",
        "provider": "deterministic fake OpenAI-compatible server",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="checkpoint-case-study-") as raw_dir:
            work = Path(raw_dir)
            db_path = work / "cassettes.db"
            daemon = run_daemon(db_path, fake.url, port=0)
            try:
                parent_id = _create_run(db_path, "diffguardian", "RECORD")
                fake_llm.reset()
                returncode, wall_ms, stdout, stderr = _spawn(
                    target,
                    daemon.url,
                    parent_id,
                    "record",
                    None,
                    work / "ledger.jsonl",
                    work / "bench.json",
                )
                parent = _finish_run(db_path, parent_id, returncode)
                if returncode != 0:
                    raise RuntimeError(f"parent failed: {stderr[-2000:]}")
                results["parent"] = {
                    **parent,
                    "provider_calls": fake_llm.call_count(),
                    "wall_ms": round(wall_ms, 2),
                    "stdout_tail": stdout[-1000:],
                }
                fake_llm.reset()

                command = subprocess.list2cmdline(target.command)
                spec = ExperimentSpec(
                    name="diffguardian-security-review",
                    fork_at=1,
                    command=command,
                    variants=[
                        Variant(name="baseline"),
                        Variant(
                            name="security_review",
                            env=dict(target.env),
                            llm_overrides={
                                "llm_call": LlmOverride(
                                    system_prompt_replace=SECURITY_PROMPT
                                )
                            },
                        ),
                    ],
                    checks=[
                        CheckSpec(type="no_errors"),
                        CheckSpec(
                            type="final_output_matches",
                            regex="SQL injection",
                        ),
                    ],
                )
                for variant in spec.variants:
                    variant.env["GROQ_API_KEY"] = "fake-groq-key"
                    variant.env["PYTHONPATH"] = str(BACKEND_ROOT)
                    variant.env["DIFFGUARDIAN_ROOT"] = str(
                        args.diffguardian_root.resolve()
                    )
                experiment = run_experiment(
                    db_path,
                    daemon.url,
                    parent_id,
                    spec,
                    base_dir=BACKEND_ROOT,
                )
                results["experiment"] = experiment
                results["post_fork_provider_calls"] = fake_llm.call_count()
                results["verdicts"] = {
                    item["variant_name"]: {
                        "passed": item["passed"],
                        "checks": [
                            {
                                "label": check["label"],
                                "passed": check["passed"],
                                "reasoning": check["reasoning"],
                            }
                            for check in item["checks"]
                        ],
                    }
                    for item in experiment["variants"]
                }
                sources = {}
                conn = init_db(db_path)
                try:
                    recorder = Recorder(conn, BlobStore(conn))
                    for item in experiment["variants"]:
                        sources[item["variant_name"]] = [
                            {
                                "step_index": int(step["step_index"]),
                                "name": step["name"],
                                "source": step["source"],
                            }
                            for step in recorder.steps_for(item["run_id"])
                        ]
                finally:
                    conn.close()
                results["step_sources"] = sources
                expected = {"baseline": False, "security_review": True}
                actual = {
                    name: verdict["passed"] for name, verdict in results["verdicts"].items()
                }
                if actual != expected:
                    raise RuntimeError(f"unexpected case-study verdicts: {actual}")
                if results["post_fork_provider_calls"] != 2:
                    raise RuntimeError(
                        "expected one live post-fork provider call per variant, got "
                        f"{results['post_fork_provider_calls']}"
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
