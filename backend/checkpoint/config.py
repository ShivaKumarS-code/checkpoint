"""Loads checkpoint.config.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

import msgspec

from checkpoint.matching.normalize import DEFAULT_MASKS, MaskRule
from checkpoint.matching.redact import DEFAULT_REDACTION, RedactionRule

DEFAULT_CONFIG_FILENAME = "checkpoint.config.toml"

MissPolicy = Literal["strict", "live"]


class ReplayConfig(msgspec.Struct, forbid_unknown_fields=True):
    """What to do when a replayed boundary has no matching recorded step."""

    llm_miss_policy: MissPolicy = "strict"
    tool_miss_policy: MissPolicy = "strict"
    checkpoint_miss_policy: MissPolicy = "strict"


class SemanticConfig(msgspec.Struct, forbid_unknown_fields=True):
    """Opt-in semantic matching, scoped to named LLM boundaries.

    Matching is always disabled unless ``enabled`` is true and the current step
    name is listed in ``step_names``. The matcher first uses a local embedding
    cosine score. When ``judge`` is true, a small OpenAI-compatible model makes
    the final equivalence decision.
    """

    enabled: bool = False
    step_names: list[str] = msgspec.field(default_factory=list)
    threshold: float = 0.92
    judge: bool = False
    judge_model: str = "gpt-4o-mini"


class Tier2Config(msgspec.Struct, forbid_unknown_fields=True):
    """Default model and limits for Tier 2 rubric checks.

    A check can override ``model``. Any OpenAI-compatible upstream works,
    including a local model server.
    """

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512


class Config(msgspec.Struct, forbid_unknown_fields=True):
    """User configuration. Unknown keys are rejected so typos fail loudly."""

    masks: list[MaskRule] = msgspec.field(default_factory=list)
    redaction: list[RedactionRule] = msgspec.field(default_factory=list)
    upstream_url: str | None = None
    freeze_clock: bool = True
    replay: ReplayConfig = msgspec.field(default_factory=ReplayConfig)
    semantic: SemanticConfig = msgspec.field(default_factory=SemanticConfig)
    tier2: Tier2Config = msgspec.field(default_factory=Tier2Config)


def validate_config(config: Config) -> None:
    """Reject semantic settings that would silently do the wrong thing."""
    if not 0.0 <= config.semantic.threshold <= 1.0:
        raise ValueError("semantic.threshold must be between 0 and 1")
    if config.semantic.enabled and not config.semantic.step_names:
        raise ValueError("semantic.enabled requires at least one entry in semantic.step_names")
    if config.semantic.judge and not config.semantic.judge_model.strip():
        raise ValueError("semantic.judge_model must not be empty when semantic.judge is true")
    if config.tier2.max_tokens < 1:
        raise ValueError("tier2.max_tokens must be >= 1")


def load_config(path: str | Path | None = None) -> Config:
    """Load config from a path, or ./checkpoint.config.toml; empty Config when absent."""
    if path is None:
        candidate = Path(DEFAULT_CONFIG_FILENAME)
        if not candidate.is_file():
            return Config()
        path = candidate
    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)
    config = msgspec.convert(data, Config)
    validate_config(config)
    return config


def effective_masks(config: Config) -> tuple[MaskRule, ...]:
    """Built-in masks plus any extra masks from the config file."""
    return DEFAULT_MASKS + tuple(config.masks)


def effective_redaction(config: Config) -> tuple[RedactionRule, ...]:
    """Built-in redaction rules plus any extra rules from the config file."""
    return DEFAULT_REDACTION + tuple(config.redaction)
