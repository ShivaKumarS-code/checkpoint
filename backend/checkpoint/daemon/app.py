"""FastAPI daemon factory and background runner. Binds to 127.0.0.1 only."""

from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from checkpoint.config import Config, effective_masks, effective_redaction, load_config
from checkpoint.daemon import events, experiments, proxy_router, query_router, sdk_router
from checkpoint.daemon.recorder import Recorder
from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import init_db

DEFAULT_HOST = "127.0.0.1"
DEFAULT_UPSTREAM_URL = "http://127.0.0.1:4141"
DEFAULT_UI_DIR = Path(__file__).resolve().parent.parent / "ui_dist"


def create_app(
    db_path: str | Path,
    upstream_url: str,
    config: Config | None = None,
    ui_dir: str | Path | None = None,
) -> FastAPI:
    if config is None:
        config = load_config()
    ui_path = (Path(ui_dir) if ui_dir is not None else DEFAULT_UI_DIR).resolve()
    index_html = ui_path / "index.html"
    has_ui = index_html.is_file()
    conn = init_db(db_path)
    blob_store = BlobStore(conn)
    bus = events.EventBus()
    recorder = Recorder(
        conn,
        blob_store,
        effective_masks(config),
        effective_redaction(config),
        on_event=bus.publish,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        bus.bind()
        app.state.http = httpx.AsyncClient(timeout=120.0)
        yield
        await app.state.http.aclose()
        conn.close()

    app = FastAPI(title="checkpoint daemon", lifespan=lifespan)
    app.state.events = bus
    app.state.recorder = recorder
    app.state.blob_store = blob_store
    app.state.config = config
    app.state.db_path = Path(db_path)
    app.state.upstream_url = upstream_url.rstrip("/")
    app.include_router(proxy_router.router)
    app.include_router(sdk_router.router)
    app.include_router(experiments.router)
    app.include_router(query_router.router)

    if has_ui and (ui_path / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=ui_path / "assets"), name="ui-assets")

    @app.get("/", response_model=None)
    async def index() -> FileResponse | dict[str, str]:
        if has_ui:
            return FileResponse(index_html)
        return {
            "name": "checkpoint",
            "docs": "/docs",
            "runs": "/api/runs",
            "events": "/ws",
        }

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str) -> FileResponse:
        if (
            not has_ui
            or full_path.startswith(("api/", "v1/", "docs", "openapi.json", "redoc"))
        ):
            raise HTTPException(status_code=404)
        candidate = (ui_path / full_path).resolve()
        if candidate.is_file() and candidate.is_relative_to(ui_path):
            return FileResponse(candidate)
        if "." in Path(full_path).name:
            raise HTTPException(status_code=404)
        return FileResponse(index_html)

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
