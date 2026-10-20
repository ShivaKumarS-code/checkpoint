"""Tier 3 swapped-order pairwise judging.

A pairwise check compares a candidate run against a reference run. The judge
sees the two runs in both orders; a verdict only counts as a win when the judge
selects the same run in both presentations. Every directional judge call is
content-addressed and cached in the existing ``judge_cache`` table, so repeated
experiments do not spend additional judge calls.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from checkpoint.evaluators.deterministic import (
    CheckResult,
    CheckSpec,
    check_label,
)
from checkpoint.matching.normalize import canonical_json

Judge = Any
PairwiseWinner = Literal["a", "b", "tie"]


@dataclass(frozen=True)
class PairwiseJudgement:
    """The result of judging the same pair in both orders."""

    winner: PairwiseWinner | None
    position_consistent: bool
    score: float | None
    reasoning: dict[str, Any]
    model: str


def pairwise_prompt(
    rubric: str, view_a: Mapping[str, Any], view_b: Mapping[str, Any]
) -> str:
    """Build one deterministic directional judging prompt."""
    return (
        "You are comparing two AI agent runs for a pairwise evaluation. "
        "The runs are presented in an arbitrary order; do not let presentation "
        "order influence your decision. Judge only the supplied material.\n\n"
        f"RUBRIC:\n{rubric}\n\n"
        f"RUN A:\n{_render_run(view_a)}\n\n"
        f"RUN B:\n{_render_run(view_b)}\n\n"
        'Return exactly one JSON object of the form '
        '{"winner": "A"|"B"|"tie", "reason": "short explanation"}. '
        'Use "tie" when the rubric does not favor either run.'
    )


def judge_pairwise(
    conn: sqlite3.Connection,
    view_a: Mapping[str, Any],
    view_b: Mapping[str, Any],
    rubric: str,
    judge: Judge | None,
    *,
    model: str,
) -> PairwiseJudgement:
    """Judge A vs B and B vs A, using and populating the judge cache.

    ``a`` and ``b`` in the returned winner refer to the first and second run
    passed to this function, independent of the order used for each call.
    """
    rubric = rubric.strip()
    model = model.strip() or "gpt-4o-mini"
    first_prompt = pairwise_prompt(rubric, view_a, view_b)
    second_prompt = pairwise_prompt(rubric, view_b, view_a)

    try:
        first_raw, first_cached = _cached_judge(conn, first_prompt, model, judge)
    except Exception as exc:
        return _failed_judgement(model, rubric, f"judge call failed: {type(exc).__name__}: {exc}")
    try:
        second_raw, second_cached = _cached_judge(conn, second_prompt, model, judge)
    except Exception as exc:
        return _failed_judgement(model, rubric, f"judge call failed: {type(exc).__name__}: {exc}")

    first_choice, first_reason = parse_pairwise_verdict(first_raw)
    second_choice, second_reason = parse_pairwise_verdict(second_raw)
    first_mapped = _map_choice(first_choice, "a_vs_b")
    second_mapped = _map_choice(second_choice, "b_vs_a")

    position_consistent = (
        first_mapped is not None
        and second_mapped is not None
        and first_mapped == second_mapped
    )
    winner = first_mapped if position_consistent else None
    score = {"a": 1.0, "b": 0.0, "tie": 0.5}.get(winner or "")
    reasoning = {
        "rubric": rubric,
        "model": model,
        "winner": winner,
        "position_consistent": position_consistent,
        "orders": [
            {
                "order": "a_vs_b",
                "choice": first_choice,
                "mapped_winner": first_mapped,
                "reason": first_reason,
                "cached": first_cached,
                "judge_response": first_raw,
            },
            {
                "order": "b_vs_a",
                "choice": second_choice,
                "mapped_winner": second_mapped,
                "reason": second_reason,
                "cached": second_cached,
                "judge_response": second_raw,
            },
        ],
    }
    return PairwiseJudgement(
        winner=winner,
        position_consistent=position_consistent,
        score=score,
        reasoning=reasoning,
        model=model,
    )


def evaluate_pairwise(
    conn: sqlite3.Connection,
    view_a: Mapping[str, Any],
    view_b: Mapping[str, Any],
    check: CheckSpec,
    judge: Judge | None,
    *,
    default_model: str = "gpt-4o-mini",
) -> CheckResult:
    """Evaluate a pairwise check from the perspective of run A (candidate)."""
    model = (check.model or default_model).strip() or default_model
    rubric = (check.rubric or "").strip()
    judgement = judge_pairwise(
        conn,
        view_a,
        view_b,
        rubric,
        judge,
        model=model,
    )
    passed = (
        judgement.position_consistent
        and (
            judgement.winner == "a"
            or (judgement.winner == "tie" and check.allow_tie)
        )
    )
    reasoning = dict(judgement.reasoning)
    if not judgement.position_consistent and "error" not in reasoning:
        reasoning["error"] = (
            "judge selected different runs when the order was swapped; "
            "the pairwise verdict is not reliable"
        )
    return CheckResult(
        check_type="pairwise",
        label=check_label(check),
        passed=passed,
        required=check.required,
        reasoning=reasoning,
        tier="TIER3",
        score=judgement.score,
        judge_model=judgement.model,
        compared_run_id=str(view_b["run"]["run_id"]),
        position_consistent=judgement.position_consistent,
    )


def parse_pairwise_verdict(raw: str) -> tuple[PairwiseWinner | None, str]:
    """Parse a pairwise judge response, accepting the common small-model shapes."""
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
        value = payload.get("winner")
        if value is None:
            value = payload.get("choice")
        if value is None:
            value = payload.get("verdict")
        choice = _normalize_choice(value)
        reason = str(payload.get("reason") or payload.get("explanation") or "")
        return choice, reason

    match = re.search(r"\b(run\s+)?(a|b|tie)\b", text, re.IGNORECASE)
    if match:
        return _normalize_choice(match.group(2)), text[:500]
    return None, "judge returned no parseable pairwise verdict"


def _normalize_choice(value: Any) -> PairwiseWinner | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"a", "run a", "first", "first run"}:
        return "a"
    if normalized in {"b", "run b", "second", "second run"}:
        return "b"
    if normalized in {"tie", "equal", "draw", "neither"}:
        return "tie"
    return None


def _map_choice(choice: str | None, order: str) -> PairwiseWinner | None:
    """Map a local A/B choice back to the candidate/reference orientation."""
    if choice == "tie":
        return "tie"
    if choice == "a":
        return "a" if order == "a_vs_b" else "b"
    if choice == "b":
        return "b" if order == "a_vs_b" else "a"
    return None


def _cached_judge(
    conn: sqlite3.Connection,
    prompt: str,
    model: str,
    judge: Judge | None,
) -> tuple[str, bool]:
    cache_key = hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()
    cached = conn.execute(
        "SELECT response_json FROM judge_cache WHERE cache_key = ?", (cache_key,)
    ).fetchone()
    if cached is not None:
        return str(cached["response_json"]), True
    if judge is None:
        raise ValueError("no Tier 3 judge configured")
    raw = str(judge(prompt, model))
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO judge_cache "
            "(cache_key, model, prompt_hash, response_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (cache_key, model, prompt_hash, raw, time.time()),
        )
    return raw, False


def _failed_judgement(model: str, rubric: str, error: str) -> PairwiseJudgement:
    return PairwiseJudgement(
        winner=None,
        position_consistent=False,
        score=None,
        reasoning={
            "rubric": rubric,
            "model": model,
            "winner": None,
            "position_consistent": False,
            "error": error,
        },
        model=model,
    )


def _render_run(view: Mapping[str, Any]) -> str:
    run = view.get("run") or {}
    steps = view.get("steps") or []
    lines = [
        f"run: {run.get('name')} (variant={run.get('variant_name') or 'none'})",
        f"final output: {_pretty(view.get('final_output'))}",
        "steps:",
    ]
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        lines.append(
            f"- {step.get('step_index')} {step.get('step_type')} {step.get('name')} "
            f"(source={step.get('source')}): {_short(step.get('output'))}"
        )
    if len(lines) == 3:
        lines.append("(none)")
    return "\n".join(lines)


def _short(value: Any, limit: int = 3000) -> str:
    rendered = _pretty(value)
    return rendered if len(rendered) <= limit else rendered[:limit] + "...<truncated>"


def _pretty(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return canonical_json(value).decode("utf-8")
    except (TypeError, ValueError):
        return repr(value)
