"""Tests for serving the built React UI from the daemon."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from checkpoint.daemon.app import create_app


def _write_ui(root: Path) -> Path:
    ui = root / "ui_dist"
    assets = ui / "assets"
    assets.mkdir(parents=True)
    (ui / "index.html").write_text(
        "<!doctype html><html><body><div id=\"root\">checkpoint ui</div></body></html>",
        encoding="utf-8",
    )
    (assets / "app-abc123.js").write_text("console.log('checkpoint ui');", encoding="utf-8")
    return ui


@pytest.fixture()
async def ui_client(tmp_path: Path) -> Any:
    ui = _write_ui(tmp_path)
    app = create_app(
        tmp_path / "cassettes.db", "http://127.0.0.1:1", ui_dir=ui
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.state.recorder._conn.close()


async def test_serves_index_assets_and_spa_fallback(ui_client: httpx.AsyncClient) -> None:
    root = await ui_client.get("/")
    assert root.status_code == 200
    assert root.headers["content-type"].startswith("text/html")
    assert "checkpoint ui" in root.text

    asset = await ui_client.get("/assets/app-abc123.js")
    assert asset.status_code == 200
    assert "console.log" in asset.text

    for path in ("/runs/abc", "/experiments/def", "/diff/abc/def"):
        response = await ui_client.get(path)
        assert response.status_code == 200
        assert "checkpoint ui" in response.text


async def test_api_and_docs_are_not_shadowed(ui_client: httpx.AsyncClient) -> None:
    assert (await ui_client.get("/api/health")).json() == {"status": "ok"}
    assert (await ui_client.get("/api/nope")).status_code == 404
    assert (await ui_client.get("/docs")).status_code == 200
    assert "/api/runs" in (await ui_client.get("/openapi.json")).json()["paths"]


async def test_root_returns_json_without_a_build(tmp_path: Path) -> None:
    app = create_app(
        tmp_path / "cassettes.db",
        "http://127.0.0.1:1",
        ui_dir=tmp_path / "missing-ui",
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            root = await client.get("/")
            assert root.headers["content-type"].startswith("application/json")
            assert root.json()["name"] == "checkpoint"
            assert (await client.get("/runs/abc")).status_code == 404
    finally:
        app.state.recorder._conn.close()
