"""Tier 2 rubric checks backed by a configurable small/local model.

The judged prompt is content-addressed and cached in SQLite. Re-running the
same check against an unchanged run therefore makes one judge call, regardless
of how many times the evaluator or experiment runner is invoked.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Callable, Mapping
from typing import Any

from checkpoint.evaluators.deterministic import CheckResult, CheckSpec, check_label
from checkpoint.matching.normalize import canonical_json

Judge = Callable[[str, str], str]


def rubric_prompt(view: Mapping[str, Any], rubric: str) -> str:
    """Build a deterministic grading prompt from a materialized run view."""
    steps = view.get("steps") or []
    step_lines: list[str] = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        step_lines.append(
            f"- step {step.get('step_index')} {step.get('step_type')} "
            f"{step.get('name')} (source={step.get('source')}): "
            f"{_short(step.get('output'))}"
        )
    return (
        "You are grading an AI agent run against a rubric. Judge only the "
        "provided run, not your assumptions about the agent.\n\n"
        f"RUBRIC:\n{rubric}\n\n"
        f"FINAL OUTPUT:\n{_pretty(view.get('final_output'))}\n\n"
        f"AGENT STEPS:\n{chr(10).join(step_lines) if step_lines else '(none)'}\n\n"
        'Return exactly one JSON object of the form '
        '{"passed": true|false, "score": 0.0-1.0, "reason": "short explanation"}.'
    )


def evaluate_rubric(
    conn: sqlite3.Connection,
    view: Mapping[str, Any],
    check: CheckSpec,
    judge: Judge | None,
    *,
    default_model: str = "gpt-4o-mini",
) -> CheckResult:
    """Evaluate one rubric check, consulting and updating the judge cache."""
    rubric = (check.rubric or "").strip()
    label = check_label(check)
    model = (check.model or default_model).strip() or default_model
    prompt = rubric_prompt(view, rubric)
    cache_key = _cache_key(model, prompt)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    cached = conn.execute(
        "SELECT response_json FROM judge_cache WHERE cache_key = ?", (cache_key,)
    ).fetchone()
    from_cache = cached is not None
    if cached is not None:
        raw = str(cached["response_json"])
    else:
        if judge is None:
            return CheckResult(
                check_type="rubric",
                label=label,
                passed=False,
                required=check.required,
                reasoning={
                    "error": "no Tier 2 judge configured; set tier2.model and an upstream",
                    "model": model,
                    "rubric": rubric,
                },
                tier="TIER2",
                judge_model=model,
            )
        try:
            raw = str(judge(prompt, model))
        except Exception as exc:  # pragma: no cover - defensive boundary
            return CheckResult(
                check_type="rubric",
                label=label,
                passed=False,
                required=check.required,
                reasoning={
                    "error": f"judge call failed: {type(exc).__name__}: {exc}",
                    "model": model,
                    "rubric": rubric,
                },
                tier="TIER2",
                judge_model=model,
            )
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO judge_cache "
                "(cache_key, model, prompt_hash, response_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (cache_key, model, prompt_hash, raw, time.time()),
            )

    passed, score, reason = parse_rubric_verdict(raw)
    return CheckResult(
        check_type="rubric",
        label=label,
        passed=passed,
        required=check.required,
        reasoning={
            "rubric": rubric,
            "model": model,
            "score": score,
            "reason": reason,
            "cached": from_cache,
            "judge_response": raw,
        },
        tier="TIER2",
        score=score,
        judge_model=model,
    )


def parse_rubric_verdict(raw: str) -> tuple[bool, float | None, str]:
    """Parse a small model's JSON verdict, rejecting anything ambiguous."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    payload: Any = None
    try:
        payload = json.loads(text)
    except ValueError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
            except ValueError:
                payload = None
    if isinstance(payload, Mapping):
        value = payload.get("passed")
        if value is None:
            value = payload.get("pass")
        if value is None:
            value = payload.get("result")
        if isinstance(value, bool):
            passed = value
        elif isinstance(value, str):
            passed = value.strip().lower() in {"pass", "passed", "true", "yes"}
        else:
            passed = False
        score_raw = payload.get("score")
        score = (
            float(score_raw)
            if isinstance(score_raw, (int, float)) and not isinstance(score_raw, bool)
            else None
        )
        reason = str(payload.get("reason") or payload.get("explanation") or "")
        return passed, score, reason

    match = re.search(r"\b(passed|pass|failed|fail)\b", text, re.IGNORECASE)
    if match:
        word = match.group(1).lower()
        return word in {"passed", "pass"}, None, text[:500]
    return False, None, "judge returned no parseable pass/fail verdict"


def _cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()


def _short(value: Any, limit: int = 4000) -> str:
    rendered = _pretty(value)
    return rendered if len(rendered) <= limit else rendered[:limit] + "...<truncated>"


def _pretty(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return canonical_json(value).decode("utf-8")
    except (TypeError, ValueError):
        return repr(value)
