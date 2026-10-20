"""Tier 2 rubric checks and judge verdict caching."""

from __future__ import annotations

from pathlib import Path

import pytest

from checkpoint.config import Config, Tier2Config
from checkpoint.daemon import experiments
from checkpoint.daemon.experiments import ExperimentSpec, SpawnResult, Variant, run_experiment
from checkpoint.daemon.recorder import Recorder
from checkpoint.evaluators.deterministic import CheckSpec, run_checks, validate_checks
from checkpoint.evaluators.rubric import parse_rubric_verdict
from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import init_db


def test_same_rubric_check_twice_makes_one_judge_call(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "cassettes.db")
    try:
        blobs = BlobStore(conn)
        recorder = Recorder(conn, blobs)
        run = recorder.create_run("rubric-run", "RECORD")
        run_id = str(run["run_id"])
        recorder.record_llm_step(
            run_id, "compose", {"task": "compose"}, {"text": "Refund issued."}, 1.0
        )
        recorder.finish_run(run_id, "COMPLETED")

        calls: list[tuple[str, str]] = []

        def judge(prompt: str, model: str) -> str:
            calls.append((prompt, model))
            return '{"passed": true, "score": 0.95, "reason": "clear and correct"}'

        check = CheckSpec(
            type="rubric",
            name="reply-quality",
            rubric="The final reply must be a clear refund confirmation.",
            model="small-local-model",
        )
        first = run_checks(
            conn, blobs, run_id, [check], judge=judge, judge_model="default-model"
        )
        second = run_checks(
            conn, blobs, run_id, [check], judge=judge, judge_model="default-model"
        )
    finally:
        conn.close()

    assert len(calls) == 1
    assert calls[0][1] == "small-local-model"
    assert first[0].passed is True
    assert first[0].tier == "TIER2"
    assert first[0].score == pytest.approx(0.95)
    assert first[0].reasoning["cached"] is False
    assert second[0].reasoning["cached"] is True
    assert second[0].judge_model == "small-local-model"


def test_judge_cache_is_persisted(tmp_path: Path) -> None:
    db_path = tmp_path / "cassettes.db"
    conn = init_db(db_path)
    blobs = BlobStore(conn)
    recorder = Recorder(conn, blobs)
    run = recorder.create_run("cache-run", "RECORD")
    run_id = str(run["run_id"])
    recorder.record_llm_step(run_id, "plan", {"task": "plan"}, {"order_id": "ORD-1"}, 1.0)
    check = CheckSpec(type="rubric", rubric="The output must contain an order id.")
    calls = 0

    def judge(prompt: str, model: str) -> str:
        nonlocal calls
        calls += 1
        return '{"passed": true, "reason": "contains an id"}'

    run_checks(conn, blobs, run_id, [check], judge=judge)
    conn.close()

    conn = init_db(db_path)
    try:
        blobs = BlobStore(conn)
        rows = conn.execute("SELECT * FROM judge_cache").fetchall()
        assert len(rows) == 1
        result = run_checks(
            conn, blobs, run_id, [check], judge=judge, judge_model="gpt-4o-mini"
        )
    finally:
        conn.close()
    assert calls == 1
    assert result[0].passed is True
    assert result[0].reasoning["cached"] is True


def test_rubric_requires_a_rubric() -> None:
    with pytest.raises(ValueError, match="rubric"):
        validate_checks([CheckSpec(type="rubric", name="empty")])


def test_parse_rubric_verdict_rejects_ambiguous_text() -> None:
    assert parse_rubric_verdict('{"passed": true, "score": 1}')[0] is True
    assert parse_rubric_verdict('```json\n{"passed": false, "reason": "wrong"}\n```')[0] is False
    assert parse_rubric_verdict("I think this is fine")[0] is False


def test_run_experiment_records_tier2_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "cassettes.db"
    conn = init_db(db_path)
    try:
        parent = Recorder(conn, BlobStore(conn)).create_run("parent", "RECORD")
        parent_id = str(parent["run_id"])
    finally:
        conn.close()

    monkeypatch.setattr(
        experiments, "_spawn", lambda command, cwd, env: SpawnResult(0, "", "")
    )

    def judge(prompt: str, model: str) -> str:
        return '{"passed": true, "score": 1.0, "reason": "good"}'

    spec = ExperimentSpec(
        fork_at=0,
        command="noop",
        variants=[Variant(name="v1")],
        checks=[CheckSpec(type="rubric", name="quality", rubric="Output is acceptable.")],
    )
    result = run_experiment(
        db_path,
        "http://127.0.0.1:1",
        parent_id,
        spec,
        base_dir=tmp_path,
        judge=judge,
        config=Config(tier2=Tier2Config(model="small-local-judge")),
    )
    variant = result["variants"][0]
    assert variant["passed"] is True
    assert variant["checks"][0]["tier"] == "TIER2"
    assert variant["checks"][0]["score"] == pytest.approx(1.0)

    conn = init_db(db_path)
    try:
        verdict = conn.execute(
            "SELECT * FROM verdicts WHERE run_id = ?", (variant["run_id"],)
        ).fetchone()
    finally:
        conn.close()
    assert verdict["tier"] == "TIER2"
    assert verdict["judge_model"] == "small-local-judge"
