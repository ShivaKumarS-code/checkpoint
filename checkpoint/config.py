"""Loads checkpoint.config.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

import msgspec

from checkpoint.matching.normalize import DEFAULT_MASKS, MaskRule

DEFAULT_CONFIG_FILENAME = "checkpoint.config.toml"


class Config(msgspec.Struct, forbid_unknown_fields=True):
    """User configuration. Unknown keys are rejected so typos fail loudly."""

    masks: list[MaskRule] = msgspec.field(default_factory=list)


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
