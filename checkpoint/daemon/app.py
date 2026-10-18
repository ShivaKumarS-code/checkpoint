"""FastAPI daemon factory and background runner. Binds to 127.0.0.1 only."""

from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI

from checkpoint.config import Config, effective_masks, effective_redaction, load_config
from checkpoint.daemon import proxy_router, sdk_router
from checkpoint.daemon.recorder import Recorder
from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import init_db

DEFAULT_HOST = "127.0.0.1"
DEFAULT_UPSTREAM_URL = "http://127.0.0.1:4141"


def create_app(
    db_path: str | Path, upstream_url: str, config: Config | None = None
) -> FastAPI:
    if config is None:
        config = load_config()
    conn = init_db(db_path)
    blob_store = BlobStore(conn)
    recorder = Recorder(
        conn, blob_store, effective_masks(config), effective_redaction(config)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.http = httpx.AsyncClient(timeout=120.0)
        yield
        await app.state.http.aclose()
        conn.close()

    app = FastAPI(title="checkpoint daemon", lifespan=lifespan)
    app.state.recorder = recorder
    app.state.blob_store = blob_store
    app.state.upstream_url = upstream_url.rstrip("/")
    app.include_router(proxy_router.router)
    app.include_router(sdk_router.router)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


class RunningDaemon:
    def __init__(self, app: FastAPI, host: str = DEFAULT_HOST, port: int = 0) -> None:
        self._app = app
        self._host = host
        self._port = port
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def start(self) -> RunningDaemon:
        config = uvicorn.Config(
            self._app, host=self._host, port=self._port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, name="checkpoint-daemon", daemon=True
        )
        self._thread.start()
        deadline = time.monotonic() + 10.0
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError("checkpoint daemon did not start")
            time.sleep(0.01)
        sockname = self._server.servers[0].sockets[0].getsockname()
        self._port = int(sockname[1])
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        self._server = None


def run_daemon(
    db_path: str | Path,
    upstream_url: str,
    config: Config | None = None,
    host: str = DEFAULT_HOST,
    port: int = 0,
) -> RunningDaemon:
    """Start the daemon in a background thread."""
    app = create_app(db_path, upstream_url, config)
    return RunningDaemon(app, host, port).start()
