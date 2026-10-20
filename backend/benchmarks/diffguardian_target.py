"""Run the real DiffGuardian LangGraph workflow under checkpoint.

This adapter is intentionally outside the default test suite: DiffGuardian's
LangGraph/Groq dependencies are optional and are not installed with checkpoint.
Set ``DIFFGUARDIAN_ROOT`` to a checkout of the DiffGuardian project, then run
this file as the child command of ``checkpoint run``/``checkpoint replay``.

External GitHub calls are replaced with deterministic fixtures. No GitHub App
private key, webhook secret, or Groq key is read; the fake provider is selected
through ``GROQ_API_BASE``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

DIFF = """diff --git a/app/auth.py b/app/auth.py
index 1111111..2222222 100644
--- a/app/auth.py
+++ b/app/auth.py
@@ -8,6 +8,10 @@ def login(request):
     username = request.args.get("username")
     password = request.args.get("password")
+    query = f"SELECT * FROM users WHERE username = '{username}'"
+    row = db.execute(query).fetchone()
+    if row and row["password"] == password:
+        return "ok"
     return "denied"
"""

FILE_CONTENT = """def login(request):
    username = request.args.get("username")
    password = request.args.get("password")
    query = f"SELECT * FROM users WHERE username = '{username}'"
    row = db.execute(query).fetchone()
    if row and row["password"] == password:
        return "ok"
    return "denied"
"""


def _disable_dotenv() -> None:
    """Keep DiffGuardian's import-time load_dotenv from reading .env files."""
    stub = types.ModuleType("dotenv")
    stub.load_dotenv = lambda *args, **kwargs: False  # type: ignore[attr-defined]
    sys.modules["dotenv"] = stub


def _parse_review(result: dict[str, Any], posted: dict[str, Any]) -> dict[str, Any]:
    messages = result.get("messages") or []
    text = ""
    if messages:
        text = str(getattr(messages[-1], "content", "") or "")
    payload: Any = None
    try:
        payload = json.loads(text)
    except ValueError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
            except ValueError:
                payload = None
    review = payload if isinstance(payload, dict) else {}
    return {
        "status": result.get("status"),
        "summary": review.get("summary", text[:1000]),
        "inline_comments": review.get("inline_comments", []),
        "posted_body": posted.get("body"),
    }


async def _main() -> int:
    root = os.environ.get("DIFFGUARDIAN_ROOT")
    if not root:
        raise SystemExit("DIFFGUARDIAN_ROOT is required")
    root_path = Path(root).resolve()
    if not (root_path / "app" / "services" / "graph.py").is_file():
        raise SystemExit(f"not a DiffGuardian checkout: {root_path}")
    sys.path.insert(0, str(root_path))

    _disable_dotenv()
    os.environ.setdefault("GITHUB_APP_ID", "1")
    os.environ.setdefault("GITHUB_PRIVATE_KEY_PATH", "not-read.pem")
    os.environ.setdefault("GITHUB_WEBHOOK_SECRET", "not-read")
    os.environ.setdefault("GROQ_API_KEY", "fake-groq-key")
    daemon_url = os.environ.get("CHECKPOINT_DAEMON_URL")
    if daemon_url:
        os.environ["GROQ_API_BASE"] = daemon_url.rstrip("/")

    import checkpoint

    checkpoint.init(run_name="diffguardian")

    import app.services.github as github

    posted: dict[str, Any] = {}

    async def fake_token(installation_id: int) -> str:
        del installation_id
        return "fake-github-token"

    async def fake_diff(repo_name: str, pr_number: int, access_token: str) -> str:
        del repo_name, pr_number, access_token
        return DIFF

    async def fake_file(
        repo_name: str, path: str, ref: str, access_token: str
    ) -> str:
        del repo_name, path, ref, access_token
        return FILE_CONTENT

    async def fake_post(
        repo_name: str,
        pr_number: int,
        access_token: str,
        commit_id: str,
        comments: list[dict[str, Any]],
        body: str,
    ) -> dict[str, Any]:
        del repo_name, pr_number, access_token, commit_id
        posted.update({"comments": comments, "body": body})
        return {"id": 1}

    # Graph imports these names directly; patch before importing the graph.
    github.get_installation_access_token = fake_token
    github.fetch_pr_diff = fake_diff
    github.fetch_file_from_repo = fake_file
    github.post_pr_comment = fake_post

    from app.services.graph import review_graph

    initial = {
        "repo_name": "acme/shop",
        "pr_number": 7,
        "installation_id": 1,
        "head_sha": "abc123",
        "token": None,
        "diff": None,
        "messages": [],
        "fetched_files": {},
        "summary": None,
        "review_comments": None,
        "status": "queued",
    }
    started = time.perf_counter()
    result = await review_graph.ainvoke(initial)
    agent_ms = (time.perf_counter() - started) * 1000.0
    state = _parse_review(result, posted)
    checkpoint.checkpoint("after_review", state)
    output = os.environ.get("CHECKPOINT_BENCHMARK_FILE")
    if output:
        with open(output, "w", encoding="utf-8") as handle:
            json.dump({"agent_ms": agent_ms, "summary": state.get("summary")}, handle)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
