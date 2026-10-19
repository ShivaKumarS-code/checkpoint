"""Thread-safe in-process event bus for WebSocket subscribers."""

from __future__ import annotations

import asyncio
import threading
from typing import Any


class EventBus:
    """Fan out daemon events to every connected WebSocket client."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = threading.Lock()

    def bind(self) -> None:
        self._loop = asyncio.get_running_loop()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        with self._lock:
            self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subscribers.discard(queue)

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            subscribers = tuple(self._subscribers)
            loop = self._loop
        if loop is None or loop.is_closed():
            return
        for queue in subscribers:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                continue

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)
