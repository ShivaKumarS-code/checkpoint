"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from examples import fake_llm


@pytest.fixture(scope="module")
def fake_llm_server() -> Iterator[fake_llm.FakeLLMServer]:
    server = fake_llm.serve()
    yield server
    server.stop()
