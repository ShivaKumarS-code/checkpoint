"""msgspec Structs mirroring the cassette schema (SPEC.md section 12)."""

from __future__ import annotations

from typing import Literal

import msgspec

RunMode = Literal["RECORD", "REPLAY", "FORK"]
RunStatus = Literal["RUNNING", "COMPLETED", "FAILED", "DIVERGED"]
StepType = Literal["LLM_CALL", "TOOL_CALL", "CHECKPOINT"]
ToolMode = Literal["pure", "stub", "live"]
StepSource = Literal["RECORDED", "REPLAYED", "LIVE", "SEMANTIC"]
VerdictTier = Literal["TIER1", "TIER2", "TIER3"]


class Run(msgspec.Struct):
    run_id: str
    name: str
    parent_run_id: str | None
    experiment_id: str | None
    variant_name: str | None
    forked_at_step: int | None
    mode: RunMode
    run_seed: int
    git_commit_sha: str | None
    status: RunStatus
    total_tokens: int
    total_cost_usd: float
    total_latency_ms: float
    divergence_step: int | None
    divergence_diff_hash: str | None
    created_at: float


class Experiment(msgspec.Struct):
    experiment_id: str
    name: str
    parent_run_id: str
    fork_step_index: int
    variants_hash: str
    checks_hash: str | None
    created_at: float


class Step(msgspec.Struct):
    step_id: str
    run_id: str
    step_index: int
    step_type: StepType
    name: str
    occurrence: int
    match_key: str
    input_hash: str
    output_hash: str | None
    pre_state_hash: str | None
    post_state_hash: str | None
    frozen_timestamp: str
    rng_seed: int
    tool_mode: ToolMode | None
    source: StepSource
    error: str | None
    latency_ms: float


class Verdict(msgspec.Struct):
    verdict_id: str
    run_id: str
    compared_run_id: str | None
    step_index: int | None
    tier: VerdictTier
    metric_name: str
    passed: bool
    score: float | None
    position_consistent: int | None
    reasoning_json: str
    judge_model: str | None
    created_at: float


class BlobRecord(msgspec.Struct):
    blob_hash: str
    compression: str
    byte_size: int
    payload: bytes
    created_at: float
