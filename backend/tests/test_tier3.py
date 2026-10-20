"""Tier 3 swapped-order pairwise judging and experiment persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from checkpoint.config import Config, Tier2Config
from checkpoint.daemon import experiments
from checkpoint.daemon.experiments import ExperimentSpec, SpawnResult, Variant, run_experiment
from checkpoint.daemon.recorder import Recorder
from checkpoint.evaluators.deterministic import CheckSpec, load_run_view
from checkpoint.evaluators.judge import evaluate_pairwise, parse_pairwise_verdict
from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import init_db


def _record_run(conn, text: str, name: str = "run") -> str:
    blobs = BlobStore(conn)
    recorder = Recorder(conn, blobs)
    run = recorder.create_run(name, "RECORD")
    run_id = str(run["run_id"])
    recorder.record_llm_step(run_id, "final", {"task": "final"}, {"text": text}, 1.0)
    recorder.finish_run(run_id, "COMPLETED")
    return run_id


def _check(**kwargs):
    return CheckSpec(type="pairwise", name="quality", rubric="Prefer the better run.", **kwargs)


def test_consistent_swapped_order_wins_and_caches_calls(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "cassettes.db")
    try:
        blobs = BlobStore(conn)
        candidate = _record_run(conn, "CANDIDATE_FIXED", "candidate")
        reference = _record_run(conn, "PARENT", "parent")
        calls: list[str] = []

        def judge(prompt: str, model: str) -> str:
            calls.append(prompt)
            return (
                '{"winner": "A", "reason": "candidate is better"}'
                if len(calls) % 2 == 1
                else '{"winner": "B", "reason": "candidate is better"}'
            )

        candidate_view = load_run_view(conn, blobs, candidate)
        reference_view = load_run_view(conn, blobs, reference)
        first = evaluate_pairwise(
            conn,
            candidate_view,
            reference_view,
            _check(model="small-judge"),
            judge,
        )
        second = evaluate_pairwise(
            conn,
            candidate_view,
            reference_view,
            _check(model="small-judge"),
            judge,
        )
    finally:
        conn.close()

    assert len(calls) == 2
    assert first.passed is True
    assert first.tier == "TIER3"
    assert first.score == pytest.approx(1.0)
    assert first.position_consistent is True
    assert first.compared_run_id == reference
    assert first.reasoning["orders"][0]["cached"] is False
    assert first.reasoning["orders"][1]["cached"] is False
    assert second.passed is True
    assert second.reasoning["orders"][0]["cached"] is True
    assert second.reasoning["orders"][1]["cached"] is True


def test_position_bias_is_inconsistent_and_fails(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "cassettes.db")
    calls = 0
    try:
        blobs = BlobStore(conn)
        candidate = _record_run(conn, "CANDIDATE", "candidate")
        reference = _record_run(conn, "PARENT", "parent")

        def judge(prompt: str, model: str) -> str:
            nonlocal calls
            calls += 1
            return '{"winner": "A", "reason": "always the first one"}'

        result = evaluate_pairwise(
            conn,
            load_run_view(conn, blobs, candidate),
            load_run_view(conn, blobs, reference),
            _check(),
            judge,
        )
    finally:
        conn.close()

    assert calls == 2
    assert result.passed is False
    assert result.position_consistent is False
    assert result.score is None
    assert result.reasoning["winner"] is None
    assert "different runs" in result.reasoning["error"]


def test_tie_is_only_a_pass_when_allow_tie_is_set(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "cassettes.db")
    try:
        blobs = BlobStore(conn)
        candidate = _record_run(conn, "CANDIDATE", "candidate")
        reference = _record_run(conn, "PARENT", "parent")

        def judge(prompt: str, model: str) -> str:
            return '{"winner": "tie", "reason": "equivalent"}'

        default = evaluate_pairwise(
            conn,
            load_run_view(conn, blobs, candidate),
            load_run_view(conn, blobs, reference),
            _check(),
            judge,
        )
        allowed = evaluate_pairwise(
            conn,
            load_run_view(conn, blobs, candidate),
            load_run_view(conn, blobs, reference),
            _check(allow_tie=True),
            judge,
        )
    finally:
        conn.close()

    assert default.passed is False
    assert default.position_consistent is True
    assert default.score == pytest.approx(0.5)
    assert allowed.passed is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"winner": "A"}', "a"),
        ('{"choice": "run B", "reason": "better"}', "b"),
        ('{"verdict": "tie"}', "tie"),
        ("The winner is B because it is shorter.", "b"),
        ("no verdict here", None),
    ],
)
def test_parse_pairwise_verdict(raw: str, expected: str | None) -> None:
    assert parse_pairwise_verdict(raw)[0] == expected


def test_pairwise_reference_resolves_parent_variant_and_run_id(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "cassettes.db")
    try:
        recorder = Recorder(conn, BlobStore(conn))
        parent = str(recorder.create_run("parent", "RECORD")["run_id"])
        fixed = str(recorder.create_run("fixed", "FORK", parent_run_id=parent)["run_id"])
        results = [{"variant_name": "fixed", "run_id": fixed}]
        assert experiments._resolve_reference(conn, parent, results, None) == (parent, None)
        assert experiments._resolve_reference(conn, parent, results, "fixed") == (fixed, None)
        assert experiments._resolve_reference(conn, parent, results, fixed) == (fixed, None)
        reference, error = experiments._resolve_reference(conn, parent, results, "missing")
        assert reference is None
        assert error == "unknown pairwise reference: missing"
    finally:
        conn.close()


def test_experiment_records_tier3_verdict_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "cassettes.db"
    conn = init_db(db_path)
    try:
        parent_id = _record_run(conn, "PARENT", "parent")
    finally:
        conn.close()

    def fake_spawn(command: str, cwd: str, env: dict[str, str]) -> SpawnResult:
        child_conn = init_db(db_path)
        try:
            blobs = BlobStore(child_conn)
            recorder = Recorder(child_conn, blobs)
            child_id = env["CHECKPOINT_RUN_ID"]
            variant = env["CHECKPOINT_VARIANT"]
            text = "CANDIDATE_FIXED" if variant == "fixed" else "CANDIDATE_BAD"
            recorder.record_llm_step(
                child_id, "final", {"task": "final"}, {"text": text}, 1.0
            )
            recorder.finish_run(child_id, "COMPLETED")
        finally:
            child_conn.close()
        return SpawnResult(0, "", "")

    monkeypatch.setattr(experiments, "_spawn", fake_spawn)
    calls: list[str] = []

    def judge(prompt: str, model: str) -> str:
        calls.append(prompt)
        a_part, b_part = prompt.split("RUN A:", 1)[1].split("RUN B:", 1)
        if "CANDIDATE_FIXED" in a_part:
            winner = "A"
        elif "CANDIDATE_FIXED" in b_part:
            winner = "B"
        elif "CANDIDATE_BAD" in a_part:
            winner = "B"
        elif "CANDIDATE_BAD" in b_part:
            winner = "A"
        else:
            winner = "tie"
        return f'{{"winner": "{winner}", "reason": "deterministic test judge"}}'

    spec = ExperimentSpec(
        fork_at=0,
        command="noop",
        variants=[Variant(name="baseline"), Variant(name="fixed")],
        checks=[_check(against="parent")],
    )
    first = run_experiment(
        db_path,
        "http://127.0.0.1:1",
        parent_id,
        spec,
        base_dir=tmp_path,
        judge=judge,
        config=Config(tier2=Tier2Config(model="small-judge")),
    )
    second = run_experiment(
        db_path,
        "http://127.0.0.1:1",
        parent_id,
        spec,
        base_dir=tmp_path,
        judge=judge,
        config=Config(tier2=Tier2Config(model="small-judge")),
    )

    assert len(calls) == 4
    by_name = {variant["variant_name"]: variant for variant in first["variants"]}
    assert by_name["fixed"]["passed"] is True
    assert by_name["baseline"]["passed"] is False
    fixed_check = by_name["fixed"]["checks"][-1]
    assert fixed_check["tier"] == "TIER3"
    assert fixed_check["compared_run_id"] == parent_id
    assert fixed_check["position_consistent"] is True
    assert fixed_check["judge_model"] == "small-judge"
    assert all(
        variant["passed"] is False
        for variant in second["variants"]
        if variant["variant_name"] == "baseline"
    )

    conn = init_db(db_path)
    try:
        baseline_id = next(
            variant["run_id"]
            for variant in first["variants"]
            if variant["variant_name"] == "baseline"
        )
        verdict = conn.execute(
            "SELECT * FROM verdicts WHERE run_id = ? AND tier = 'TIER3'",
            (baseline_id,),
        ).fetchone()
    finally:
        conn.close()
    assert verdict is not None
    assert verdict["compared_run_id"] == parent_id
    assert verdict["position_consistent"] == 1
