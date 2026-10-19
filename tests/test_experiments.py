"""End-to-end experiment tests: demo fork, overrides, checks, concurrency, CLI."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from checkpoint.daemon import experiments
from checkpoint.daemon.experiments import (
    ExperimentSpec,
    SpawnResult,
    Variant,
    load_experiment_spec,
    run_experiment,
)
from checkpoint.daemon.recorder import Recorder
from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import init_db
from examples import fake_llm

from .conftest import DaemonHandle
from .test_proxy_record import create_run
from .test_replay import steps_for

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "examples" / "refund_agent"
TICKET = "Please refund order ORD-1002. It arrived damaged."


def record_parent(daemon: DaemonHandle, ledger: Path) -> str:
    parent = create_run(daemon.url, name="refund-agent")
    env = os.environ.copy()
    env.update(
        {
            "CHECKPOINT_DAEMON_URL": daemon.url,
            "CHECKPOINT_RUN_ID": parent,
            "CHECKPOINT_MODE": "record",
            "OPENAI_BASE_URL": f"{daemon.url}/v1",
            "REFUND_AGENT_LEDGER": str(ledger),
        }
    )
    completed = subprocess.run(
        [sys.executable, "agent.py", TICKET],
        cwd=DEMO_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    conn = init_db(daemon.db_path)
    try:
        Recorder(conn, BlobStore(conn)).finish_run(parent, "COMPLETED")
    finally:
        conn.close()
    return parent


def demo_spec(ledger: Path) -> ExperimentSpec:
    spec = load_experiment_spec(DEMO_DIR / "variants.yaml")
    spec.command = f'"{sys.executable}" agent.py'
    for variant in spec.variants:
        variant.env.setdefault("REFUND_AGENT_LEDGER", str(ledger))
    return spec


def test_fork_experiment_fixes_the_bug_and_counts_upstream_calls(
    daemon: DaemonHandle, fake_llm_server: fake_llm.FakeLLMServer, tmp_path: Path
) -> None:
    parent_ledger = tmp_path / "parent_ledger.jsonl"
    parent = record_parent(daemon, parent_ledger)
    assert parent_ledger.exists()
    ledger = tmp_path / "fork_ledger.jsonl"
    ledger.unlink(missing_ok=True)
    fake_llm.reset()

    spec = demo_spec(ledger)
    result = run_experiment(
        daemon.db_path, daemon.url, parent, spec, base_dir=DEMO_DIR
    )
    by_name = {variant["variant_name"]: variant for variant in result["variants"]}
    assert set(by_name) == {"baseline", "cheaper_model", "guard_prior_refund"}
    assert by_name["guard_prior_refund"]["passed"] is True
    assert by_name["baseline"]["passed"] is False
    assert by_name["cheaper_model"]["passed"] is False

    baseline_failures = {
        check["label"] for check in by_name["baseline"]["checks"] if not check["passed"]
    }
    assert baseline_failures == {
        "tool_not_called:issue_refund",
        "state_equals:after_refund:refund",
        "final_output_matches",
    }
    guard_failures = [
        check["label"] for check in by_name["guard_prior_refund"]["checks"] if not check["passed"]
    ]
    assert guard_failures == []

    # Only the compose step (index >= fork) may reach the upstream; the plan
    # and decide calls are replayed from the cassette for every variant.
    assert fake_llm.call_count() == 3
    for variant in by_name.values():
        steps = steps_for(daemon.db_path, variant["run_id"])
        sources = {step["name"]: step["source"] for step in steps}
        assert sources["plan"] == "REPLAYED"
        assert sources["decide"] == "REPLAYED"
        assert sources["compose"] == "LIVE"

    # Stubs never execute in any variant, so the fork ledger is never written.
    assert ledger.exists() is False
    assert not any(
        step["name"] == "issue_refund" for step in steps_for(
            daemon.db_path, by_name["guard_prior_refund"]["run_id"]
        )
    )
    baseline_refund = next(
        step
        for step in steps_for(daemon.db_path, by_name["baseline"]["run_id"])
        if step["name"] == "issue_refund"
    )
    assert baseline_refund["source"] == "REPLAYED"

    # The cheaper_model variant applied its model override to the live call.
    cheaper_compose = next(
        step
        for step in steps_for(daemon.db_path, by_name["cheaper_model"]["run_id"])
        if step["name"] == "compose"
    )
    conn = init_db(daemon.db_path)
    try:
        compose_input = BlobStore(conn).get_json(cheaper_compose["input_hash"])
        assert compose_input["model"] == "gpt-4o-mini"
        assert len(Recorder(conn, BlobStore(conn)).verdicts_for_run(
            by_name["baseline"]["run_id"]
        )) == 4
    finally:
        conn.close()

    experiment = httpx_get(daemon.url, f"/api/experiments/{result['experiment_id']}")
    assert experiment["experiment"]["fork_step_index"] == 5
    assert len(experiment["runs"]) == 3
    assert len(experiment["variants"]) == 3


def httpx_get(url: str, path: str) -> dict:
    import httpx

    response = httpx.get(f"{url}{path}", timeout=30.0)
    response.raise_for_status()
    return response.json()


def test_twenty_variants_never_exceed_the_concurrency_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "cassettes.db"
    conn = init_db(db_path)
    try:
        parent = Recorder(conn, BlobStore(conn)).create_run("parent", "RECORD")
        parent_id = str(parent["run_id"])
    finally:
        conn.close()

    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_spawn(command: str, cwd: str, env: dict[str, str]) -> SpawnResult:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return SpawnResult(0, "", "")

    monkeypatch.setattr(experiments, "_spawn", fake_spawn)
    spec = ExperimentSpec(
        fork_at=0,
        command="noop",
        variants=[Variant(name=f"v{index}") for index in range(20)],
        checks=[],
    )
    result = run_experiment(
        db_path, "http://127.0.0.1:1", parent_id, spec, base_dir=tmp_path, max_parallel=4
    )
    assert result["max_parallel"] == 4
    assert len(result["variants"]) == 20
    assert max_active == 4
    assert all(variant["passed"] for variant in result["variants"])

    conn = init_db(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM runs WHERE experiment_id = ?", (result["experiment_id"],)
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 20
    assert {row["mode"] for row in rows} == {"FORK"}
    assert {row["forked_at_step"] for row in rows} == {0}


def _cli_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    scripts = str(Path(sys.executable).parent)
    env["PATH"] = scripts + os.pathsep + env.get("PATH", "")
    for key in list(env):
        if key.startswith("CHECKPOINT_"):
            env.pop(key)
    if extra:
        env.update(extra)
    return env


def _checkpoint_exe() -> str:
    candidate = Path(sys.executable).with_name("checkpoint.exe")
    return str(candidate) if candidate.is_file() else "checkpoint"


def test_checkpoint_test_exit_codes(
    daemon: DaemonHandle, fake_llm_server: fake_llm.FakeLLMServer, tmp_path: Path
) -> None:
    parent = record_parent(daemon, tmp_path / "parent_ledger.jsonl")
    node = "'" + f'"{sys.executable}" agent.py' + "'"
    cwd = f"'{DEMO_DIR}'"
    failing_yaml = tmp_path / "failing.yaml"
    failing_yaml.write_text(
        f"fork_at: 5\n"
        f"command: {node}\n"
        f"variants:\n"
        f"  - name: baseline\n"
        f"    cwd: {cwd}\n"
        f"checks:\n"
        f"  - type: tool_not_called\n"
        f"    name: issue_refund\n",
        encoding="utf-8",
    )
    passing_yaml = tmp_path / "passing.yaml"
    passing_yaml.write_text(
        f"fork_at: 5\n"
        f"command: {node}\n"
        f"variants:\n"
        f"  - name: guard_prior_refund\n"
        f"    cwd: {cwd}\n"
        f"    env:\n"
        f"      REFUND_AGENT_GUARD_PRIOR: '1'\n"
        f"      REFUND_AGENT_LEDGER: '{tmp_path / 'guard_ledger.jsonl'}'\n"
        f"checks:\n"
        f"  - type: tool_not_called\n"
        f"    name: issue_refund\n"
        f"  - type: state_equals\n"
        f"    checkpoint: after_refund\n"
        f"    path: refund\n"
        f"    value: null\n",
        encoding="utf-8",
    )

    base_args = [
        _checkpoint_exe(),
        "test",
        "--run",
        parent,
        "--db",
        str(daemon.db_path),
        "--port",
        "0",
        "--upstream",
        fake_llm_server.url,
    ]
    failing = subprocess.run(
        [*base_args, str(failing_yaml)],
        cwd=REPO_ROOT,
        env=_cli_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert failing.returncode == 1, failing.stderr
    assert "baseline failed checks:" in failing.stdout
    assert "tool_not_called:issue_refund" in failing.stdout

    passing = subprocess.run(
        [*base_args, str(passing_yaml)],
        cwd=REPO_ROOT,
        env=_cli_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert passing.returncode == 0, passing.stdout + passing.stderr
    assert "guard_prior_refund" in passing.stdout
    assert "pass" in passing.stdout
