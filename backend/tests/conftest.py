"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from checkpoint.daemon.app import run_daemon
from checkpoint.sdk import client as sdk_client
from examples import fake_llm


@pytest.fixture(scope="module")
def fake_llm_server() -> Iterator[fake_llm.FakeLLMServer]:
    server = fake_llm.serve()
    yield server
    server.stop()


@dataclass
class DaemonHandle:
    url: str
    db_path: Path


@pytest.fixture()
def daemon(fake_llm_server: fake_llm.FakeLLMServer, tmp_path: Path) -> Iterator[DaemonHandle]:
    db_path = tmp_path / "cassettes.db"
    running = run_daemon(db_path, fake_llm_server.url, port=0)
    yield DaemonHandle(url=running.url, db_path=db_path)
    running.stop()


@pytest.fixture(autouse=True)
def reset_sdk_runtime() -> Iterator[None]:
    yield
    sdk_client.reset()
