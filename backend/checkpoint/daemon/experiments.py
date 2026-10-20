"""Experiment definitions, variants.yaml parsing, and the bounded-concurrency runner.

A fork experiment replays the parent's steps before the fork point from the
cassette and runs the variant's agent from there. Each variant gets its own
child run linked by ``parent_run_id``/``forked_at_step``/``experiment_id``.
"""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import msgspec
import yaml
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from checkpoint.config import Config
from checkpoint.daemon.recorder import Recorder
from checkpoint.evaluators.deterministic import (
    CheckResult,
    CheckSpec,
    check_label,
    load_run_view,
    run_checks,
    validate_checks,
)
from checkpoint.evaluators.judge import evaluate_pairwise
from checkpoint.evaluators.rubric import Judge
from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import init_db

router = APIRouter()


class LlmOverride(msgspec.Struct, forbid_unknown_fields=True):
    """Request rewrites applied to live LLM calls at or after the fork point."""

    model: str | None = None
    temperature: float | None = None
    system_prompt_replace: str | None = None
    messages_patch: list[dict[str, Any]] | None = None


class Variant(msgspec.Struct, forbid_unknown_fields=True):
    name: str
    command: str | None = None
    cwd: str | None = None
    env: dict[str, str] = msgspec.field(default_factory=dict)
    llm_overrides: dict[str, LlmOverride] = msgspec.field(default_factory=dict)


class ExperimentSpec(msgspec.Struct, forbid_unknown_fields=True):
    run_id: str | None = None
    name: str | None = None
    fork_at: int | None = None
    fork_at_step_name: str | None = None
    command: str | None = None
    variants: list[Variant] = msgspec.field(default_factory=list)
    checks: list[CheckSpec] = msgspec.field(default_factory=list)
    max_parallel: int = 4


class ExperimentRequest(msgspec.Struct, forbid_unknown_fields=True):
    parent_run_id: str
    spec: ExperimentSpec
    base_dir: str | None = None
    max_parallel: int | None = None


@dataclass(frozen=True)
class SpawnResult:
    returncode: int
    stdout: str
    stderr: str


def _spawn(command: str, cwd: str, env: dict[str, str]) -> SpawnResult:
    completed = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
    )
    return SpawnResult(completed.returncode, completed.stdout, completed.stderr)


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    """Load, validate and path-resolve a variants.yaml file."""
    yaml_path = Path(path).resolve()
    if not yaml_path.is_file():
        raise ValueError(f"variants file not found: {yaml_path}")
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("variants file must contain a mapping")
    try:
        spec = msgspec.convert(dict(raw), ExperimentSpec)
    except msgspec.ValidationError as exc:
        raise ValueError(f"invalid variants file: {exc}") from exc
    validate_spec(spec, yaml_path.parent)
    return spec


def validate_spec(spec: ExperimentSpec, base_dir: str | Path) -> None:
    base = Path(base_dir).resolve()
    if (spec.fork_at is None) == (spec.fork_at_step_name is None):
        raise ValueError("exactly one of fork_at or fork_at_step_name is required")
    if spec.fork_at is not None and spec.fork_at < 0:
        raise ValueError("fork_at must be >= 0")
    if spec.fork_at_step_name is not None and not spec.fork_at_step_name.strip():
        raise ValueError("fork_at_step_name must not be empty")
    if spec.max_parallel < 1:
        raise ValueError("max_parallel must be >= 1")
    if not spec.variants:
        raise ValueError("at least one variant is required")
    names: set[str] = set()
    for variant in spec.variants:
        if not variant.name.strip():
            raise ValueError("variant names must not be empty")
        if variant.name in names:
            raise ValueError(f"duplicate variant name: {variant.name}")
        names.add(variant.name)
        if not (variant.command or spec.command):
            raise ValueError(f"variant {variant.name} has no command and no top-level command")
        if variant.cwd is not None:
            cwd = Path(variant.cwd)
            if not cwd.is_absolute():
                cwd = base / cwd
            cwd = cwd.resolve()
            if not cwd.is_dir():
                raise ValueError(f"variant {variant.name} cwd does not exist: {cwd}")
            variant.cwd = str(cwd)
    validate_checks(spec.checks)


def resolve_fork_step(conn: Any, parent_run_id: str, spec: ExperimentSpec) -> int:
    parent = conn.execute("SELECT * FROM runs WHERE run_id = ?", (parent_run_id,)).fetchone()
    if parent is None:
        raise ValueError(f"unknown parent run: {parent_run_id}")
    count = int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM steps WHERE run_id = ?", (parent_run_id,)
        ).fetchone()["n"]
    )
    if spec.fork_at_step_name is not None:
        row = conn.execute(
            "SELECT step_index FROM steps WHERE run_id = ? AND name = ? "
            "ORDER BY step_index LIMIT 1",
            (parent_run_id, spec.fork_at_step_name),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"no step named {spec.fork_at_step_name!r} in run {parent_run_id}"
            )
        fork_index = int(row["step_index"])
    else:
        fork_index = int(spec.fork_at or 0)
    if fork_index > count:
        raise ValueError(
            f"fork step {fork_index} is beyond the parent's {count} recorded steps"
        )
    return fork_index


def _variant_env(
    daemon_url: str,
    parent_run_id: str,
    child_run_id: str,
    fork_step: int,
    variant: Variant,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CHECKPOINT_DAEMON_URL": daemon_url.rstrip("/"),
            "CHECKPOINT_RUN_ID": child_run_id,
            "CHECKPOINT_MODE": "fork",
            "CHECKPOINT_PARENT_RUN_ID": parent_run_id,
            "CHECKPOINT_FORK_AT": str(fork_step),
            "CHECKPOINT_VARIANT": variant.name,
            "OPENAI_BASE_URL": f"{daemon_url.rstrip('/')}/v1",
            "ANTHROPIC_BASE_URL": f"{daemon_url.rstrip('/')}/v1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.update(variant.env)
    return env


def _upstream_judge(
    upstream_url: str, *, temperature: float, max_tokens: int
) -> Judge:
    """Create a Tier 2 judge backed by the OpenAI-compatible upstream."""
    api_key = os.environ.get("CHECKPOINT_JUDGE_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def judge(prompt: str, model: str) -> str:
        response = httpx.post(
            f"{upstream_url.rstrip('/')}/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a strict rubric grader. Reply with JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=120.0,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") if isinstance(payload, Mapping) else None
        first = choices[0] if isinstance(choices, list) and choices else None
        message = first.get("message") if isinstance(first, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise ValueError("judge upstream returned no text content")
        return content

    return judge


def run_experiment(
    db_path: str | Path,
    daemon_url: str,
    parent_run_id: str,
    spec: ExperimentSpec,
    base_dir: str | Path | None = None,
    max_parallel: int | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    config: Config | None = None,
    upstream_url: str | None = None,
    judge: Judge | None = None,
) -> dict[str, Any]:
    """Run every variant and return the experiment plus per-variant verdicts.

    Each worker opens its own SQLite connection so the runner never shares a
    connection with the daemon's request handlers.
    """
    base = Path(base_dir or Path.cwd()).resolve()
    validate_spec(spec, base)
    tier2_model = config.tier2.model if config is not None else "gpt-4o-mini"
    tier2_temperature = config.tier2.temperature if config is not None else 0.0
    tier2_max_tokens = config.tier2.max_tokens if config is not None else 512
    judge_fn = judge
    if judge_fn is None and upstream_url:
        judge_fn = _upstream_judge(
            upstream_url,
            temperature=tier2_temperature,
            max_tokens=tier2_max_tokens,
        )
    single_checks = [check for check in spec.checks if check.type != "pairwise"]

    conn = init_db(db_path)
    try:
        recorder = Recorder(conn, BlobStore(conn), on_event=on_event)
        parent = recorder.get_run(parent_run_id)
        if parent is None:
            raise ValueError(f"unknown parent run: {parent_run_id}")
        fork_step = resolve_fork_step(conn, parent_run_id, spec)
        experiment_name = spec.name or f"fork:{parent['name']}"
        variants_builtin = [msgspec.to_builtins(variant) for variant in spec.variants]
        checks_builtin = [msgspec.to_builtins(check) for check in spec.checks]
        experiment_id = recorder.create_experiment(
            experiment_name,
            parent_run_id,
            fork_step,
            variants_builtin,
            checks_builtin,
        )
        parent_seed = int(parent["run_seed"])
        parent_name = str(parent["name"])
    finally:
        conn.close()

    limit = max_parallel if max_parallel is not None else spec.max_parallel
    limit = max(1, min(limit, len(spec.variants)))
    semaphore = threading.Semaphore(limit)

    def run_variant(variant: Variant) -> dict[str, Any]:
        worker_conn = init_db(db_path)
        try:
            worker_blobs = BlobStore(worker_conn)
            worker = Recorder(worker_conn, worker_blobs, on_event=on_event)
            created = worker.create_run(
                parent_name,
                "FORK",
                parent_run_id=parent_run_id,
                forked_at_step=fork_step,
                variant_name=variant.name,
                experiment_id=experiment_id,
                run_seed=parent_seed,
            )
            child_run_id = str(created["run_id"])
            command = variant.command or spec.command
            assert command is not None
            cwd = variant.cwd or str(base)
            env = _variant_env(daemon_url, parent_run_id, child_run_id, fork_step, variant)
            with semaphore:
                try:
                    spawn = _spawn(command, cwd, env)
                except OSError as exc:
                    spawn = SpawnResult(127, "", f"{type(exc).__name__}: {exc}")
            worker.finish_run(
                child_run_id, "COMPLETED" if spawn.returncode == 0 else "FAILED"
            )
            run = worker.get_run(child_run_id)
            status = str(run["status"]) if run is not None else "FAILED"
            checks = run_checks(
                worker_conn,
                worker_blobs,
                child_run_id,
                single_checks,
                base,
                judge=judge_fn,
                judge_model=tier2_model,
            )
            for check in checks:
                worker.record_verdict(
                    child_run_id,
                    check.label,
                    check.passed,
                    {
                        "check_type": check.check_type,
                        "required": check.required,
                        **check.reasoning,
                    },
                    tier=check.tier,
                    score=check.score,
                    judge_model=check.judge_model,
                )
            semantic_steps = [
                {
                    "step_index": int(step["step_index"]),
                    "name": str(step["name"]),
                    "occurrence": int(step["occurrence"]),
                }
                for step in worker.steps_for(child_run_id)
                if step["source"] == "SEMANTIC"
            ]
            passed = all(check.passed for check in checks if check.required)
            return {
                "variant_name": variant.name,
                "run_id": child_run_id,
                "exit_code": spawn.returncode,
                "status": status,
                "divergence_step": run["divergence_step"] if run is not None else None,
                "passed": passed,
                "checks": [_check_payload(check) for check in checks],
                "semantic_matches": len(semantic_steps),
                "semantic_steps": semantic_steps,
                "stdout": spawn.stdout,
                "stderr": spawn.stderr,
            }
        finally:
            worker_conn.close()

    order = {variant.name: index for index, variant in enumerate(spec.variants)}
    with ThreadPoolExecutor(max_workers=len(spec.variants)) as executor:
        futures = {
            executor.submit(run_variant, variant): variant for variant in spec.variants
        }
        results: list[dict[str, Any]] = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            _emit_event(
                on_event,
                {
                    "type": "experiment_progress",
                    "experiment_id": experiment_id,
                    "variant_name": result["variant_name"],
                    "run_id": result["run_id"],
                    "passed": result["passed"],
                    "semantic_matches": result["semantic_matches"],
                    "completed": len(results),
                    "total": len(spec.variants),
                },
            )
    results.sort(key=lambda result: order[str(result["variant_name"])])
    _run_pairwise_checks(
        db_path,
        parent_run_id,
        spec.checks,
        results,
        judge_fn,
        tier2_model,
    )
    for result in results:
        result["passed"] = all(
            check["passed"] for check in result["checks"] if check["required"]
        )
    semantic_steps = [
        {
            "variant_name": result["variant_name"],
            "run_id": result["run_id"],
            **step,
        }
        for result in results
        for step in result["semantic_steps"]
    ]
    _emit_event(
        on_event,
        {
            "type": "experiment_finished",
            "experiment_id": experiment_id,
            "total": len(results),
            "passed": sum(1 for result in results if result["passed"]),
            "failed": sum(1 for result in results if not result["passed"]),
            "semantic_matches": len(semantic_steps),
        },
    )

    return {
        "experiment_id": experiment_id,
        "name": experiment_name,
        "parent_run_id": parent_run_id,
        "fork_step_index": fork_step,
        "max_parallel": limit,
        "variants": results,
        "semantic_matches": len(semantic_steps),
        "semantic_steps": semantic_steps,
        "semantic_summary": _semantic_summary(len(semantic_steps)),
    }


def _emit_event(
    on_event: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]
) -> None:
    if on_event is None:
        return
    try:
        on_event(event)
    except Exception:
        pass


def _run_pairwise_checks(
    db_path: str | Path,
    parent_run_id: str,
    checks: Sequence[CheckSpec],
    results: list[dict[str, Any]],
    judge_fn: Judge | None,
    default_model: str,
) -> None:
    """Run Tier 3 checks after every variant has finished.

    A pairwise check compares each candidate variant to ``against`` (the parent
    by default, or a named variant/run id). The reference itself is skipped.
    Failures are recorded as check results rather than aborting the experiment.
    """
    pairwise_checks = [check for check in checks if check.type == "pairwise"]
    if not pairwise_checks:
        return
    conn = init_db(db_path)
    try:
        blobs = BlobStore(conn)
        recorder = Recorder(conn, blobs)
        for check in pairwise_checks:
            reference_id, resolve_error = _resolve_reference(
                conn, parent_run_id, results, check.against
            )
            for result in results:
                candidate_id = str(result["run_id"])
                if reference_id is not None and candidate_id == reference_id:
                    continue
                if resolve_error is not None or reference_id is None:
                    check_result = CheckResult(
                        check_type="pairwise",
                        label=check_label(check),
                        passed=False,
                        required=check.required,
                        reasoning={
                            "error": resolve_error or "no reference run found",
                            "against": check.against,
                        },
                        tier="TIER3",
                        judge_model=check.model or default_model,
                    )
                else:
                    try:
                        view_a = load_run_view(conn, blobs, candidate_id)
                        view_b = load_run_view(conn, blobs, reference_id)
                        check_result = evaluate_pairwise(
                            conn,
                            view_a,
                            view_b,
                            check,
                            judge_fn,
                            default_model=default_model,
                        )
                    except Exception as exc:
                        check_result = CheckResult(
                            check_type="pairwise",
                            label=check_label(check),
                            passed=False,
                            required=check.required,
                            reasoning={
                                "error": f"{type(exc).__name__}: {exc}",
                                "against": check.against,
                            },
                            tier="TIER3",
                            judge_model=check.model or default_model,
                            compared_run_id=reference_id,
                        )
                recorder.record_verdict(
                    candidate_id,
                    check_result.label,
                    check_result.passed,
                    {
                        "check_type": check_result.check_type,
                        "required": check_result.required,
                        **check_result.reasoning,
                    },
                    tier=check_result.tier,
                    score=check_result.score,
                    judge_model=check_result.judge_model,
                    compared_run_id=check_result.compared_run_id,
                    position_consistent=check_result.position_consistent,
                )
                result["checks"].append(_check_payload(check_result))
    finally:
        conn.close()


def _resolve_reference(
    conn: Any,
    parent_run_id: str,
    results: Sequence[dict[str, Any]],
    against: str | None,
) -> tuple[str | None, str | None]:
    target = (against or "parent").strip()
    if target == "parent":
        return parent_run_id, None
    for result in results:
        if result["variant_name"] == target:
            return str(result["run_id"]), None
    row = conn.execute("SELECT run_id FROM runs WHERE run_id = ?", (target,)).fetchone()
    if row is not None:
        return str(row["run_id"]), None
    return None, f"unknown pairwise reference: {target}"


def _check_payload(check: CheckResult) -> dict[str, Any]:
    return {
        "label": check.label,
        "check_type": check.check_type,
        "required": check.required,
        "passed": check.passed,
        "reasoning": check.reasoning,
        "tier": check.tier,
        "score": check.score,
        "judge_model": check.judge_model,
        "compared_run_id": check.compared_run_id,
        "position_consistent": check.position_consistent,
    }


def _semantic_summary(count: int) -> str:
    noun = "step" if count == 1 else "steps"
    return f"{count} {noun} served by semantic match"


def _experiment_payload(recorder: Recorder, experiment: Any) -> dict[str, Any]:
    experiment_id = str(experiment["experiment_id"])
    variants = recorder.experiment_variants(experiment_id)
    checks = recorder.experiment_checks(experiment_id)
    runs: list[dict[str, Any]] = []
    semantic_steps: list[dict[str, Any]] = []
    for run in recorder.experiment_runs(experiment_id):
        run_id = str(run["run_id"])
        verdicts = [
            {
                "verdict_id": verdict["verdict_id"],
                "metric_name": verdict["metric_name"],
                "tier": verdict["tier"],
                "passed": bool(verdict["passed"]),
                "score": verdict["score"],
                "judge_model": verdict["judge_model"],
                "compared_run_id": verdict["compared_run_id"],
                "position_consistent": (
                    None
                    if verdict["position_consistent"] is None
                    else bool(verdict["position_consistent"])
                ),
                "reasoning": msgspec.json.decode(str(verdict["reasoning_json"])),
            }
            for verdict in recorder.verdicts_for_run(run_id)
        ]
        run_semantic = [
            {
                "step_index": int(step["step_index"]),
                "name": str(step["name"]),
                "occurrence": int(step["occurrence"]),
            }
            for step in recorder.steps_for(run_id)
            if step["source"] == "SEMANTIC"
        ]
        semantic_steps.extend(
            {
                "variant_name": run["variant_name"],
                "run_id": run_id,
                **step,
            }
            for step in run_semantic
        )
        runs.append(
            {
                "run": dict(run),
                "verdicts": verdicts,
                "semantic_matches": len(run_semantic),
                "semantic_steps": run_semantic,
            }
        )
    return {
        "experiment": dict(experiment),
        "variants": variants,
        "checks": checks,
        "runs": runs,
        "semantic_matches": len(semantic_steps),
        "semantic_steps": semantic_steps,
        "semantic_summary": _semantic_summary(len(semantic_steps)),
    }


@router.post("/api/experiments")
async def start_experiment(request: Request) -> JSONResponse:
    """Run an experiment off the event loop so subprocesses can reach the proxy."""
    try:
        payload = msgspec.json.decode(await request.body(), type=ExperimentRequest)
    except msgspec.ValidationError as exc:
        return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
    try:
        result = await run_in_threadpool(
            run_experiment,
            db_path=request.app.state.db_path,
            daemon_url=str(request.base_url).rstrip("/"),
            parent_run_id=payload.parent_run_id,
            spec=payload.spec,
            base_dir=payload.base_dir,
            max_parallel=payload.max_parallel,
            on_event=request.app.state.events.publish,
            config=request.app.state.config,
            upstream_url=request.app.state.upstream_url,
        )
    except (ValueError, KeyError) as exc:
        return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
    return JSONResponse(result)


@router.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: str, request: Request) -> JSONResponse:
    conn = init_db(request.app.state.db_path)
    try:
        recorder = Recorder(conn, BlobStore(conn))
        experiment = recorder.get_experiment(experiment_id)
        if experiment is None:
            return JSONResponse(
                {"error": {"message": f"unknown experiment: {experiment_id}"}},
                status_code=404,
            )
        return JSONResponse(_experiment_payload(recorder, experiment))
    finally:
        conn.close()


def experiment_variant_overrides(
    recorder: Recorder, experiment_id: str | None, variant_name: str | None
) -> dict[str, Any]:
    """Resolve a run's variant ``llm_overrides`` keyed by step name."""
    if not experiment_id or not variant_name:
        return {}
    variant = recorder.experiment_variant(experiment_id, variant_name)
    if variant is None:
        return {}
    overrides = variant.get("llm_overrides")
    return dict(overrides) if isinstance(overrides, Mapping) else {}


def apply_llm_overrides(
    body: dict[str, Any], step_name: str, overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply model/temperature/system-prompt/messages overrides to a request body."""
    rule = overrides.get(step_name) or overrides.get("*")
    if not isinstance(rule, Mapping):
        return body
    updated = dict(body)
    if isinstance(rule.get("model"), str):
        updated["model"] = rule["model"]
    if isinstance(rule.get("temperature"), (int, float)):
        updated["temperature"] = rule["temperature"]
    messages = list(updated.get("messages") or [])
    replacement = rule.get("system_prompt_replace")
    if isinstance(replacement, str):
        replaced = False
        for index, message in enumerate(messages):
            if isinstance(message, Mapping) and message.get("role") == "system":
                messages[index] = {**message, "content": replacement}
                replaced = True
                break
        if not replaced:
            messages.insert(0, {"role": "system", "content": replacement})
    patch = rule.get("messages_patch")
    if isinstance(patch, Sequence):
        messages.extend(item for item in patch if isinstance(item, Mapping))
    updated["messages"] = messages
    return updated
