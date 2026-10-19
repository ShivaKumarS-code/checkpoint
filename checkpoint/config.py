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


class Config(msgspec.Struct, forbid_unknown_fields=True):
    """User configuration. Unknown keys are rejected so typos fail loudly."""

    masks: list[MaskRule] = msgspec.field(default_factory=list)
    redaction: list[RedactionRule] = msgspec.field(default_factory=list)
    upstream_url: str | None = None
    freeze_clock: bool = True
    replay: ReplayConfig = msgspec.field(default_factory=ReplayConfig)


def load_config(path: str | Path | None = None) -> Config:
    """Load config from a path, or ./checkpoint.config.toml; empty Config when absent."""
    if path is None:
        candidate = Path(DEFAULT_CONFIG_FILENAME)
        if not candidate.is_file():
            return Config()
        path = candidate
    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)
    return msgspec.convert(data, Config)


def effective_masks(config: Config) -> tuple[MaskRule, ...]:
    """Built-in masks plus any extra masks from the config file."""
    return DEFAULT_MASKS + tuple(config.masks)


def effective_redaction(config: Config) -> tuple[RedactionRule, ...]:
    """Built-in redaction rules plus any extra rules from the config file."""
    return DEFAULT_REDACTION + tuple(config.redaction)
