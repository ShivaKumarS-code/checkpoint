"""Opt-in semantic replay tests and experiment-summary reporting."""

from __future__ import annotations

from pathlib import Path

import httpx

from checkpoint.config import Config, SemanticConfig
from checkpoint.daemon import experiments
from checkpoint.daemon.app import run_daemon
from checkpoint.daemon.recorder import Recorder
from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import init_db
from examples import fake_llm

from .conftest import DaemonHandle
from .test_proxy_record import chat_body, create_run, fetch_steps, post_chat


def test_semantic_matching_is_off_by_default(daemon: DaemonHandle) -> None:
    parent = create_run(daemon.url)
    recorded = post_chat(daemon.url, chat_body("please refund ORD-1002"), run_id=parent)
    assert recorded.status_code == 200

    replay = create_run(
        daemon.url,
        name="replay",
        mode="REPLAY",
        parent_run_id=parent,
        run_seed=1,
    )
    response = post_chat(daemon.url, chat_body("please refund ORD-1003"), run_id=replay)
    assert response.status_code == 409
    assert response.json()["error"]["type"] == "checkpoint_divergence"


def test_semantic_hit_serves_recorded_output_and_marks_source(
    fake_llm_server: fake_llm.FakeLLMServer, tmp_path: Path
) -> None:
    db_path = tmp_path / "cassettes.db"
    config = Config(
        semantic=SemanticConfig(
            enabled=True,
            step_names=["plan"],
            threshold=0.8,
            judge=False,
        )
    )
    running = run_daemon(db_path, fake_llm_server.url, config, port=0)
    try:
        parent = create_run(running.url)
        body = chat_body("please refund ORD-1002")
        recorded = post_chat(running.url, body, run_id=parent, step_name="plan")
        assert recorded.status_code == 200

        fake_llm.reset()
        replay = create_run(
            running.url,
            name="replay",
            mode="REPLAY",
            parent_run_id=parent,
            run_seed=1,
        )
        changed = chat_body("please refund ORD-1003")
        response = post_chat(running.url, changed, run_id=replay, step_name="plan")
        assert response.status_code == 200
        assert response.json() == recorded.json()
        assert fake_llm.call_count() == 0

        steps = [step for step in fetch_steps(db_path) if step["run_id"] == replay]
        assert len(steps) == 1
        assert steps[0]["source"] == "SEMANTIC"
        assert steps[0]["name"] == "plan"

        # A step name that was not opted in stays strict.
        other_replay = create_run(
            running.url,
            name="replay-other",
            mode="REPLAY",
            parent_run_id=parent,
            run_seed=1,
        )
        other = post_chat(
            running.url,
            chat_body("please refund ORD-1003"),
            run_id=other_replay,
            step_name="compose",
        )
        assert other.status_code == 409
    finally:
        running.stop()


def test_optional_semantic_judge_confirms_the_hit(
    fake_llm_server: fake_llm.FakeLLMServer, tmp_path: Path
) -> None:
    db_path = tmp_path / "cassettes.db"
    config = Config(
        semantic=SemanticConfig(
            enabled=True,
            step_names=["plan"],
            threshold=0.8,
            judge=True,
            judge_model="fake-judge",
        )
    )
    running = run_daemon(db_path, fake_llm_server.url, config, port=0)
    try:
        parent = create_run(running.url)
        recorded = post_chat(
            running.url, chat_body("please refund ORD-1002"), run_id=parent, step_name="plan"
        )
        assert recorded.status_code == 200

        replay = create_run(
            running.url,
            name="replay",
            mode="REPLAY",
            parent_run_id=parent,
            run_seed=1,
        )
        fake_llm.reset()
        response = post_chat(
            running.url,
            chat_body("please refund ORD-1003"),
            run_id=replay,
            step_name="plan",
        )
        assert response.status_code == 200
        assert fake_llm.call_count() == 1
        steps = [step for step in fetch_steps(db_path) if step["run_id"] == replay]
        assert steps[0]["source"] == "SEMANTIC"
    finally:
        running.stop()


def test_experiment_payload_lists_every_semantic_hit(daemon: DaemonHandle) -> None:
    parent = create_run(daemon.url)
    post_chat(daemon.url, chat_body("semantic parent"), run_id=parent, step_name="plan")

    conn = init_db(daemon.db_path)
    try:
        recorder = Recorder(conn, BlobStore(conn))
        experiment_id = recorder.create_experiment(
            "semantic-summary",
            parent,
            0,
            [{"name": "variant-a"}],
            [],
        )
        child = recorder.create_run(
            "semantic-summary",
            "FORK",
            parent_run_id=parent,
            forked_at_step=0,
            variant_name="variant-a",
            experiment_id=experiment_id,
            run_seed=1,
        )
        child_id = str(child["run_id"])
        recorder.record_step(
            run_id=child_id,
            step_type="LLM_CALL",
            name="plan",
            match_key_value="semantic-test",
            input_obj={"prompt": "different wording"},
            output_obj={"text": "served semantically"},
            source="SEMANTIC",
        )
        experiment = recorder.get_experiment(experiment_id)
        assert experiment is not None
        payload = experiments._experiment_payload(recorder, experiment)
    finally:
        conn.close()

    assert payload["semantic_matches"] == 1
    assert payload["semantic_summary"] == "1 step served by semantic match"
    assert payload["semantic_steps"] == [
        {
            "variant_name": "variant-a",
            "run_id": child_id,
            "step_index": 0,
            "name": "plan",
            "occurrence": 0,
        }
    ]
    assert payload["runs"][0]["semantic_matches"] == 1

    response = httpx.get(f"{daemon.url}/api/experiments/{experiment_id}", timeout=10.0)
    assert response.status_code == 200
    body = response.json()
    assert body["semantic_matches"] == 1
    assert body["semantic_steps"][0]["name"] == "plan"

    listing = httpx.get(f"{daemon.url}/api/experiments", timeout=10.0)
    assert listing.status_code == 200
    summary = next(
        item
        for item in listing.json()["experiments"]
        if item["experiment_id"] == experiment_id
    )
    assert summary["semantic_matches"] == 1
    assert summary["semantic_summary"] == "1 step served by semantic match"
