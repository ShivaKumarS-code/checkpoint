"""Tests for variants.yaml parsing and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from checkpoint.daemon.experiments import (
    apply_llm_overrides,
    load_experiment_spec,
)

BASE = """
name: test-fork
fork_at: 2
command: python agent.py
variants:
  - name: baseline
  - name: wildcard
    llm_overrides:
      "*":
        model: small-model
        temperature: 0.2
        system_prompt_replace: "You must be careful."
        messages_patch:
          - role: user
            content: extra
  - name: with_env
    env:
      FEATURE_X: "1"
    cwd: work
checks:
  - type: no_errors
  - type: tool_not_called
    name: delete_order
  - type: state_equals
    checkpoint: after_refund
    path: refund
    value: null
  - type: python
    path: checks/my_checks.py::check
"""


def _write(tmp_path: Path, text: str) -> Path:
    (tmp_path / "work").mkdir(exist_ok=True)
    path = tmp_path / "variants.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_resolves_paths_and_parses_overrides(tmp_path: Path) -> None:
    spec = load_experiment_spec(_write(tmp_path, BASE))
    assert spec.name == "test-fork"
    assert spec.fork_at == 2
    assert spec.max_parallel == 4
    assert [variant.name for variant in spec.variants] == ["baseline", "wildcard", "with_env"]
    wildcard = spec.variants[1]
    override = wildcard.llm_overrides["*"]
    assert override.model == "small-model"
    assert override.temperature == pytest.approx(0.2)
    assert override.system_prompt_replace == "You must be careful."
    assert override.messages_patch == [{"role": "user", "content": "extra"}]
    assert spec.variants[2].cwd == str((tmp_path / "work").resolve())
    assert spec.checks[0].type == "no_errors"
    assert spec.checks[1].name == "delete_order"
    assert spec.checks[2].value is None


def test_fork_at_step_name_is_supported(tmp_path: Path) -> None:
    text = BASE.replace("fork_at: 2", "fork_at_step_name: decide")
    spec = load_experiment_spec(_write(tmp_path, text))
    assert spec.fork_at is None
    assert spec.fork_at_step_name == "decide"


def test_variant_command_overrides_top_level(tmp_path: Path) -> None:
    text = BASE.replace("  - name: baseline", "  - name: baseline\n    command: other.py")
    spec = load_experiment_spec(_write(tmp_path, text))
    assert spec.variants[0].command == "other.py"


@pytest.mark.parametrize(
    "text",
    [
        BASE.replace("fork_at: 2", "fork_at: 2\nfork_at_step_name: decide"),
        BASE.replace("fork_at: 2", ""),
        BASE.replace("fork_at: 2", "fork_at: -1"),
        BASE.replace("command: python agent.py", ""),
        BASE.replace("  - name: baseline", "  - name: baseline\n  - name: baseline"),
        BASE.replace("fork_at: 2", "fork_at: 2\nunknown_key: true"),
        BASE.replace("type: python", "type: not_a_check"),
        BASE.replace("path: checks/my_checks.py::check", "path: checks/my_checks.py"),
        BASE.replace("    name: delete_order", ""),
        BASE
        + "  - type: final_output_matches\n    regex: '['\n",
        BASE.replace("fork_at: 2", "fork_at: 2\nmax_parallel: 0"),
    ],
)
def test_invalid_specs_are_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValueError):
        load_experiment_spec(_write(tmp_path, text))


def test_invalid_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "variants.yaml"
    path.write_text("variants: [", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid YAML"):
        load_experiment_spec(path)


def test_apply_llm_overrides_by_name_and_wildcard() -> None:
    body = {
        "model": "original",
        "temperature": 0.0,
        "messages": [{"role": "system", "content": "old"}, {"role": "user", "content": "hi"}],
    }
    overrides = {
        "plan": {
            "model": "named-model",
            "system_prompt_replace": "new system",
            "messages_patch": [{"role": "user", "content": "patched"}],
        },
        "*": {"model": "wildcard-model", "temperature": 0.7},
    }
    named = apply_llm_overrides(body, "plan", overrides)
    assert named["model"] == "named-model"
    assert named["temperature"] == 0.0
    assert named["messages"][0]["content"] == "new system"
    assert named["messages"][-1]["content"] == "patched"
    assert body["model"] == "original"

    wildcard = apply_llm_overrides(body, "compose", overrides)
    assert wildcard["model"] == "wildcard-model"
    assert wildcard["temperature"] == 0.7

    unchanged = apply_llm_overrides(body, "compose", {"plan": {"model": "other"}})
    assert unchanged == body
