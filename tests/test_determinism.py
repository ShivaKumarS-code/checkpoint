"""Tests for deterministic clock, UUID, urandom and random handling."""

from __future__ import annotations

import os
import random
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from checkpoint.sdk import determinism

START = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)


@pytest.fixture()
def seeded() -> Iterator[determinism.Determinism]:
    controller = determinism.install(run_seed=42, clock_start=START.timestamp())
    yield controller
    determinism.stop()


def test_clock_is_frozen_at_run_start_and_advances(
    seeded: determinism.Determinism,
) -> None:
    assert datetime.now(UTC) == START
    assert datetime.now(UTC).isoformat() == START.isoformat()

    seeded.advance_to("2024-01-02T03:05:00+00:00")
    assert datetime.now(UTC) == datetime(2024, 1, 2, 3, 5, tzinfo=UTC)


def test_real_now_is_not_frozen(seeded: determinism.Determinism) -> None:
    real = datetime.fromisoformat(determinism.real_now_iso())
    assert real > START
    assert datetime.now(UTC) == START


def test_uuid_urandom_and_random_are_reproducible() -> None:
    snapshots = []
    for _ in range(2):
        determinism.install(run_seed=7, clock_start=START.timestamp())
        snapshots.append(
            {
                "uuid": [uuid.uuid4().hex for _ in range(3)],
                "urandom": [os.urandom(12).hex() for _ in range(2)],
                "random": [random.random() for _ in range(3)],
            }
        )
        determinism.stop()
    assert snapshots[0] == snapshots[1]
    assert len(set(snapshots[0]["uuid"])) == 3
    assert all(len(value) == 24 for value in snapshots[0]["urandom"])


def test_different_run_seeds_produce_different_entropy() -> None:
    determinism.install(run_seed=1, clock_start=START.timestamp())
    first = (uuid.uuid4().hex, os.urandom(8))
    determinism.stop()
    determinism.install(run_seed=2, clock_start=START.timestamp())
    second = (uuid.uuid4().hex, os.urandom(8))
    determinism.stop()
    assert first != second


def test_seed_hooks_run_at_install_and_each_boundary() -> None:
    seen: list[int] = []
    controller = determinism.install(
        run_seed=11, clock_start=START.timestamp(), seed_hooks=[seen.append]
    )
    try:
        assert seen == [determinism.boundary_seed(11, 0)]
        assert controller.begin_boundary() == 0
        assert controller.begin_boundary() == 1
        assert seen == [
            determinism.boundary_seed(11, 0),
            determinism.boundary_seed(11, 0),
            determinism.boundary_seed(11, 1),
        ]
    finally:
        determinism.stop()


def test_boundary_seed_matches_formula_and_varies() -> None:
    assert determinism.boundary_seed(5, 0) == (5 * 6364136223846793005) & 0x7FFFFFFFFFFFFFFF
    seeds = {determinism.boundary_seed(5, index) for index in range(20)}
    assert len(seeds) == 20


def test_stop_restores_the_clock_and_random_state() -> None:
    original_uuid = uuid.uuid4
    original_urandom = os.urandom
    state = random.getstate()
    determinism.install(run_seed=3, clock_start=START.timestamp())
    determinism.stop()
    assert uuid.uuid4 is original_uuid
    assert os.urandom is original_urandom
    assert random.getstate() == state
    assert datetime.now(UTC) != START


def test_freeze_clock_opt_out_leaves_time_alone() -> None:
    determinism.install(run_seed=3, clock_start=START.timestamp(), freeze_clock=False)
    try:
        assert datetime.now(UTC) != START
        first = uuid.uuid4().hex
        determinism.stop()
        determinism.install(run_seed=3, clock_start=START.timestamp(), freeze_clock=False)
        assert uuid.uuid4().hex == first
    finally:
        determinism.stop()
