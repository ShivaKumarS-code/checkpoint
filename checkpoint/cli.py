"""Command line interface: checkpoint run, show."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from checkpoint.config import Config, effective_masks, effective_redaction, load_config
from checkpoint.daemon.app import DEFAULT_UPSTREAM_URL, run_daemon
from checkpoint.daemon.recorder import Recorder
from checkpoint.storage.cas import BlobStore
from checkpoint.storage.db import connect, init_db

app = typer.Typer(
    add_completion=False,
    help="checkpoint: deterministic replay and fork debugger for AI agents",
)
console = Console()

DEFAULT_DB_PATH = Path(".checkpoint/cassettes.db")


def resolve_db(db: Path | None) -> Path:
    if db is not None:
        return db
    env = os.environ.get("CHECKPOINT_DB_PATH")
    return Path(env) if env else DEFAULT_DB_PATH


def resolve_upstream(upstream: str | None, config: Config) -> str:
    if upstream:
        return upstream
    env = os.environ.get("CHECKPOINT_UPSTREAM_URL")
    if env:
        return env
    return config.upstream_url or DEFAULT_UPSTREAM_URL


def print_steps(run_id: str, db_path: Path) -> None:
    conn = connect(db_path)
    try:
        run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if run is None:
            console.print(f"[red]unknown run:[/red] {run_id}")
            raise typer.Exit(code=1)
        console.print(
            f"run [bold]{run['run_id']}[/bold]  name={run['name']}  "
            f"mode={run['mode']}  status={run['status']}"
        )
        table = Table("step", "type", "name", "occ", "source", "latency_ms", "error")
        for row in conn.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY step_index", (run_id,)
        ):
            table.add_row(
                str(row["step_index"]),
                row["step_type"],
                row["name"],
                str(row["occurrence"]),
                row["source"],
                f"{row['latency_ms']:.1f}",
                row["error"] or "",
            )
        console.print(table)
    finally:
        conn.close()


@app.command()
def run(
    command: list[str] = typer.Argument(..., help="agent command to run in RECORD mode"),
    name: str | None = typer.Option(None, "--name", "-n", help="run name"),
    db: Path | None = typer.Option(None, "--db", help="cassette database path"),
    upstream: str | None = typer.Option(
        None, "--upstream", "-u", help="upstream OpenAI-compatible base URL"
    ),
    port: int = typer.Option(4040, "--port", help="daemon port (0 picks a free port)"),
) -> None:
    """Run an agent in RECORD mode under the local proxy."""
    config = load_config()
    db_path = resolve_db(db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    upstream_url = resolve_upstream(upstream, config)
    run_name = name or Path(command[0]).stem

    conn = init_db(db_path)
    recorder = Recorder(
        conn, BlobStore(conn), effective_masks(config), effective_redaction(config)
    )
    created = recorder.create_run(run_name, "RECORD")
    run_id = str(created["run_id"])

    daemon = run_daemon(db_path, upstream_url, config, port=port)
    env = os.environ.copy()
    env.update(
        {
            "CHECKPOINT_DAEMON_URL": daemon.url,
            "CHECKPOINT_RUN_ID": run_id,
            "CHECKPOINT_MODE": "record",
            "OPENAI_BASE_URL": f"{daemon.url}/v1",
        }
    )
    exit_code = 1
    try:
        console.print(
            f"recording run [bold]{run_id}[/bold] as [bold]{run_name}[/bold] "
            f"(upstream {upstream_url})"
        )
        executable = shutil.which(command[0]) or command[0]
        completed = subprocess.run([executable, *command[1:]], env=env)
        exit_code = completed.returncode if completed.returncode >= 0 else 1
    except FileNotFoundError as exc:
        console.print(f"[red]failed to launch:[/red] {exc}")
        exit_code = 127
    finally:
        daemon.stop()
        recorder.finish_run(run_id, "COMPLETED" if exit_code == 0 else "FAILED")
        conn.close()

    print_steps(run_id, db_path)
    if exit_code != 0:
        console.print(f"[red]agent exited with code {exit_code}; run marked FAILED[/red]")
    raise typer.Exit(code=exit_code)


@app.command()
def show(
    run_id: str = typer.Argument(..., help="run id"),
    db: Path | None = typer.Option(None, "--db", help="cassette database path"),
    step: int | None = typer.Option(None, "--step", help="show one step's payloads"),
) -> None:
    """Print a run's steps in the terminal."""
    db_path = resolve_db(db)
    if step is None:
        print_steps(run_id, db_path)
        return
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM steps WHERE run_id = ? AND step_index = ?", (run_id, step)
        ).fetchone()
        if row is None:
            console.print(f"[red]no step {step} in run[/red] {run_id}")
            raise typer.Exit(code=1)
        console.print(
            f"step {row['step_index']}  {row['step_type']}  {row['name']}  "
            f"source={row['source']}  latency_ms={row['latency_ms']:.1f}"
        )
        blobs = BlobStore(conn)
        console.print("input:")
        console.print_json(data=blobs.get_json(row["input_hash"]))
        if row["output_hash"]:
            console.print("output:")
            console.print_json(data=blobs.get_json(row["output_hash"]))
    finally:
        conn.close()


def main() -> None:
    app()
