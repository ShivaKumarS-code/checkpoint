"""Tests for `checkpoint diff` and `checkpoint web`."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from checkpoint import cli as cli_module
from checkpoint.cli import app
from checkpoint.daemon.recorder import Recorder
from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import init_db


def _run(db: Path, name: str, prompts: list[str], suffix: str = "ok") -> str:
    conn = init_db(db)
    try:
        recorder = Recorder(conn, BlobStore(conn))
        run_id = str(recorder.create_run(name, "RECORD")["run_id"])
        for index, prompt in enumerate(prompts):
            recorder.record_llm_step(
                run_id,
                f"step{index}",
                {"model": "m", "messages": [{"role": "user", "content": prompt}]},
                {"choices": [{"message": {"content": suffix}}]},
                latency_ms=1.0,
            )
        recorder.finish_run(run_id, "COMPLETED")
        return run_id
    finally:
        conn.close()


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / "cassettes.db"


def test_identical_runs_have_no_divergence(db: Path) -> None:
    left = _run(db, "a", ["one", "two"])
    right = _run(db, "b", ["one", "two"], suffix="different output")
    result = CliRunner().invoke(app, ["diff", left, right, "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "no divergent step" in result.output
    assert "2 identical step inputs" in result.output


def test_changed_prompt_points_at_the_first_differing_step(db: Path) -> None:
    left = _run(db, "a", ["one", "two"])
    right = _run(db, "b", ["one", "CHANGED"])
    result = CliRunner().invoke(app, ["diff", left, right, "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "first divergence at step 1" in result.output
    assert "step1" in result.output
    assert "CHANGED" in result.output
    assert "two" in result.output


def test_shorter_run_diverges_at_the_missing_step(db: Path) -> None:
    left = _run(db, "a", ["one", "two"])
    right = _run(db, "b", ["one"])
    result = CliRunner().invoke(app, ["diff", left, right, "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "first divergence at step 1" in result.output
    assert "<no recorded step>" in result.output


def test_unknown_run_exits_nonzero(db: Path) -> None:
    left = _run(db, "a", ["one"])
    result = CliRunner().invoke(app, ["diff", left, "missing", "--db", str(db)])
    assert result.exit_code == 1
    assert "unknown run" in result.output


def test_web_command_starts_daemon_without_blocking(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeDaemon:
        url = "http://127.0.0.1:9"

        def stop(self) -> None:
            pass

    class FakeThreading:
        class Event:
            def wait(self) -> None:
                raise KeyboardInterrupt

    opened: list[str] = []
    monkeypatch.setattr(cli_module, "run_daemon", lambda *args, **kwargs: FakeDaemon())
    monkeypatch.setattr(cli_module, "threading", FakeThreading)
    monkeypatch.setattr(cli_module.webbrowser, "open", lambda url: opened.append(url))
    result = CliRunner().invoke(
        app,
        ["web", "--db", str(db), "--port", "0"],
    )
    assert result.exit_code == 0, result.output
    assert "checkpoint daemon at" in result.output
    assert opened == ["http://127.0.0.1:9"]


def test_root_help_lists_phase_four_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "diff" in result.output
    assert "web" in result.output
