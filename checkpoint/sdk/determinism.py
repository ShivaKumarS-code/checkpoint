"""Deterministic clock, UUID, os.urandom and random handling.

Determinism is applied in both RECORD and REPLAY. Otherwise prompts built
from ``datetime.now()`` or ``uuid4()`` would differ between the two and every
match_key would miss.

Clock semantics (SPEC.md section 6): the clock is frozen at the run start
time. At each boundary it advances to the real current time (record) or to
the recorded step timestamp (replay). Code between boundaries sees the last
boundary's time. ``time.monotonic`` and ``time.perf_counter`` are left real.
"""

from __future__ import annotations

import hashlib
import os
import random
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

import time_machine

SeedHook = Callable[[int], None]


def boundary_seed(run_seed: int, step_index: int) -> int:
    """Deterministic per-boundary RNG seed, stored now and reused by replay."""
    return (run_seed * 6364136223846793005 + step_index) & 0x7FFFFFFFFFFFFFFF


def real_now_iso() -> str:
    """Wall-clock time, even while time-machine is freezing ``datetime.now``."""
    hatch: Any = time_machine.escape_hatch
    if hatch.is_travelling():
        return hatch.datetime.datetime.now(UTC).isoformat()
    return datetime.now(UTC).isoformat()


class Determinism:
    """Installs and removes the process-wide determinism patches."""

    def __init__(
        self,
        run_seed: int,
        clock_start: float,
        freeze_clock: bool = True,
        seed_hooks: Iterable[SeedHook] = (),
    ) -> None:
        self._run_seed = run_seed
        self._clock_start = clock_start
        self._freeze_clock = freeze_clock
        self._seed_hooks = tuple(seed_hooks)
        self._lock = threading.Lock()
        self._boundary_index = 0
        self._uuid_counter = 0
        self._urandom_counter = 0
        self._travel: Any = None
        self._traveller: Any = None
        self._original_uuid4 = uuid.uuid4
        self._original_urandom = os.urandom
        self._random_state: tuple[Any, ...] | None = None
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        self._random_state = random.getstate()
        if self._freeze_clock:
            start = datetime.fromtimestamp(self._clock_start, tz=UTC)
            self._travel = time_machine.travel(start, tick=False)
            self._traveller = self._travel.start()
        uuid.uuid4 = self._uuid4  # type: ignore[assignment]
        os.urandom = self._urandom  # type: ignore[assignment]
        self._seed_for(0)
        self._installed = True

    def stop(self) -> None:
        if not self._installed:
            return
        uuid.uuid4 = self._original_uuid4  # type: ignore[assignment]
        os.urandom = self._original_urandom  # type: ignore[assignment]
        if self._travel is not None:
            self._travel.stop()
        if self._random_state is not None:
            random.setstate(self._random_state)
        self._travel = None
        self._traveller = None
        self._installed = False

    def begin_boundary(self) -> int:
        """Allocate the next boundary index and reseed entropy for it."""
        with self._lock:
            index = self._boundary_index
            self._boundary_index += 1
        self._seed_for(index)
        return index

    def advance_to(self, timestamp: str | None) -> None:
        """Move the frozen clock to a boundary timestamp (replay or record)."""
        if not self._freeze_clock or self._traveller is None or timestamp is None:
            return
        self._traveller.move_to(datetime.fromisoformat(timestamp))

    def seed_for(self, step_index: int) -> None:
        self._seed_for(step_index)

    def seed_with(self, seed: int) -> None:
        """Reseed entropy with an exact seed (used for recorded step seeds)."""
        random.seed(seed)
        for hook in self._seed_hooks:
            hook(seed)

    def _seed_for(self, step_index: int) -> None:
        self.seed_with(boundary_seed(self._run_seed, step_index))

    def _uuid4(self) -> uuid.UUID:
        with self._lock:
            self._uuid_counter += 1
            counter = self._uuid_counter
        digest = hashlib.sha256(f"{self._run_seed}:uuid4:{counter}".encode("ascii")).digest()
        return uuid.UUID(bytes=digest[:16], version=4)

    def _urandom(self, size: int) -> bytes:
        with self._lock:
            self._urandom_counter += 1
            counter = self._urandom_counter
        output = bytearray()
        block = 0
        while len(output) < size:
            payload = f"{self._run_seed}:urandom:{counter}:{block}".encode("ascii")
            output.extend(hashlib.sha256(payload).digest())
            block += 1
        return bytes(output[:size])


current: Determinism | None = None


def install(
    run_seed: int,
    clock_start: float,
    freeze_clock: bool = True,
    seed_hooks: Iterable[SeedHook] = (),
) -> Determinism:
    global current
    if current is not None:
        current.stop()
    controller = Determinism(run_seed, clock_start, freeze_clock, seed_hooks)
    controller.install()
    current = controller
    return controller


def stop() -> None:
    global current
    if current is not None:
        current.stop()
        current = None


def controller() -> Determinism | None:
    return current


def monotonic_now() -> float:
    """Real monotonic time, usable for latency measurements."""
    return time.perf_counter()
