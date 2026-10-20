"""Command line interface: checkpoint run, replay, fork, test, show."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import webbrowser
from pathlib import Path
from typing import Any

import httpx
import msgspec
import typer
from rich.console import Console
from rich.table import Table

from checkpoint.config import Config, effective_masks, effective_redaction, load_config
from checkpoint.daemon.app import DEFAULT_UPSTREAM_URL, run_daemon
from checkpoint.daemon.experiments import load_experiment_spec
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
        status = run["status"]
        if run["divergence_step"] is not None:
            status = f"{status} at step {run['divergence_step']}"
        console.print(
            f"run [bold]{run['run_id']}[/bold]  name={run['name']}  "
            f"mode={run['mode']}  status={status}"
        )
        if run["parent_run_id"]:
            console.print(f"parent: {run['parent_run_id']}")
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
        if run["divergence_diff_hash"]:
            diff = BlobStore(conn).get_json(run["divergence_diff_hash"])
            if isinstance(diff, dict) and isinstance(diff.get("diff"), str):
                console.print("[red]divergence:[/red]")
                console.print(diff["diff"], markup=False)
    finally:
        conn.close()


def _launch(command: list[str], env: dict[str, str]) -> int:
    executable = shutil.which(command[0]) or command[0]
    try:
        completed = subprocess.run([executable, *command[1:]], env=env)
    except FileNotFoundError as exc:
        console.print(f"[red]failed to launch:[/red] {exc}")
        return 127
    return completed.returncode if completed.returncode >= 0 else 1


def _run_variants(
    parent_run_id: str,
    variants_path: Path,
    db: Path | None,
    upstream: str | None,
    port: int,
    max_parallel: int | None,
    require_pass: bool,
) -> int:
    spec = load_experiment_spec(variants_path)
    db_path = resolve_db(db)
    if not db_path.is_file():
        console.print(f"[red]cassette database not found:[/red] {db_path}")
        return 1
    config = load_config()
    upstream_url = resolve_upstream(upstream, config)
    daemon = run_daemon(db_path, upstream_url, config, port=port)
    try:
        response = httpx.post(
            f"{daemon.url}/api/experiments",
            json={
                "parent_run_id": parent_run_id,
                "spec": msgspec.to_builtins(spec),
                "base_dir": str(Path(variants_path).resolve().parent),
                "max_parallel": max_parallel,
            },
            timeout=None,
        )
        if response.status_code >= 400:
            message = response.text
            try:
                error = response.json().get("error")
                if isinstance(error, dict) and error.get("message"):
                    message = str(error["message"])
            except ValueError:
                pass
            console.print(f"[red]experiment failed:[/red] {message}")
            return 1
        result = response.json()
    finally:
        daemon.stop()
    _print_experiment(result)
    if require_pass and any(not variant["passed"] for variant in result["variants"]):
        return 1
    return 0


def _print_experiment(result: dict[str, Any]) -> None:
    console.print(
        f"experiment [bold]{result['experiment_id']}[/bold]  "
        f"parent={result['parent_run_id']}  fork_at={result['fork_step_index']}"
    )
    table = Table("variant", "run", "status", "checks", "result")
    for variant in result["variants"]:
        checks = variant["checks"]
        passed_checks = sum(1 for check in checks if check["passed"])
        status = variant["status"]
        if variant.get("divergence_step") is not None:
            status = f"{status} at step {variant['divergence_step']}"
        table.add_row(
            variant["variant_name"],
            variant["run_id"],
            status,
            f"{passed_checks}/{len(checks)}",
            "[green]pass[/green]" if variant["passed"] else "[red]fail[/red]",
        )
    console.print(table)
    semantic_count = int(result.get("semantic_matches") or 0)
    if semantic_count:
        summary = result.get("semantic_summary") or (
            f"{semantic_count} steps served by semantic match"
        )
        console.print(f"[green]{summary}[/green]")
        for step in result.get("semantic_steps", []):
            console.print(
                f"  [dim]{step.get('variant_name')} step {step.get('step_index')}: "
                f"{step.get('name')}[/dim]"
            )
    for variant in result["variants"]:
        failed = [check["label"] for check in variant["checks"] if not check["passed"]]
        if failed:
            console.print(
                f"[red]{variant['variant_name']} failed checks:[/red] {', '.join(failed)}"
            )
        if variant["passed"]:
            continue
        for stream in ("stdout", "stderr"):
            text = str(variant.get(stream) or "").strip()
            if text:
                console.print(f"[dim]{variant['variant_name']} {stream}:[/dim]")
                console.print(text, markup=False)


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
            "ANTHROPIC_BASE_URL": f"{daemon.url}/v1",
        }
    )
    exit_code = 1
    try:
        console.print(
            f"recording run [bold]{run_id}[/bold] as [bold]{run_name}[/bold] "
            f"(upstream {upstream_url})"
        )
        exit_code = _launch(command, env)
    finally:
        daemon.stop()
        recorder.finish_run(run_id, "COMPLETED" if exit_code == 0 else "FAILED")
        conn.close()

    print_steps(run_id, db_path)
    if exit_code != 0:
        console.print(f"[red]agent exited with code {exit_code}; run marked FAILED[/red]")
    raise typer.Exit(code=exit_code)


@app.command()
def replay(
    run_id: str = typer.Argument(..., help="recorded run id to replay"),
    command: list[str] = typer.Argument(..., help="agent command to replay"),
    db: Path | None = typer.Option(None, "--db", help="cassette database path"),
    upstream: str | None = typer.Option(
        None, "--upstream", "-u", help="upstream base URL for live misses"
    ),
    port: int = typer.Option(4040, "--port", help="daemon port (0 picks a free port)"),
) -> None:
    """Replay a recorded run. Local agent code re-executes; boundary calls come
    from the cassette. Strict miss policy by default."""
    config = load_config()
    db_path = resolve_db(db)
    if not db_path.is_file():
        console.print(f"[red]cassette database not found:[/red] {db_path}")
        raise typer.Exit(code=1)
    upstream_url = resolve_upstream(upstream, config)

    conn = init_db(db_path)
    recorder = Recorder(
        conn, BlobStore(conn), effective_masks(config), effective_redaction(config)
    )
    parent = recorder.get_run(run_id)
    if parent is None:
        console.print(f"[red]unknown run:[/red] {run_id}")
        conn.close()
        raise typer.Exit(code=1)
    created = recorder.create_run(
        str(parent["name"]),
        "REPLAY",
        parent_run_id=run_id,
        run_seed=int(parent["run_seed"]),
    )
    child_id = str(created["run_id"])

    daemon = run_daemon(db_path, upstream_url, config, port=port)
    env = os.environ.copy()
    env.update(
        {
            "CHECKPOINT_DAEMON_URL": daemon.url,
            "CHECKPOINT_RUN_ID": child_id,
            "CHECKPOINT_MODE": "replay",
            "OPENAI_BASE_URL": f"{daemon.url}/v1",
            "ANTHROPIC_BASE_URL": f"{daemon.url}/v1",
        }
    )
    exit_code = 1
    try:
        console.print(f"replaying run [bold]{run_id}[/bold] as [bold]{child_id}[/bold]")
        exit_code = _launch(command, env)
    finally:
        daemon.stop()
        recorder.finish_run(child_id, "COMPLETED" if exit_code == 0 else "FAILED")
        completed_run = recorder.get_run(child_id)
        conn.close()

    print_steps(child_id, db_path)
    if completed_run is not None and completed_run["status"] == "DIVERGED":
        console.print(f"[red]replay diverged from run {run_id}[/red]")
        raise typer.Exit(code=1)
    if exit_code != 0:
        console.print(f"[red]agent exited with code {exit_code}; replay marked FAILED[/red]")
    raise typer.Exit(code=exit_code)


@app.command()
def fork(
    run_id: str = typer.Argument(..., help="recorded run id to fork"),
    variants: Path = typer.Option(
        ..., "--variants", "-v", help="variants.yaml describing variants and checks"
    ),
    db: Path | None = typer.Option(None, "--db", help="cassette database path"),
    upstream: str | None = typer.Option(
        None, "--upstream", "-u", help="upstream OpenAI-compatible base URL"
    ),
    port: int = typer.Option(4040, "--port", help="daemon port (0 picks a free port)"),
    max_parallel: int | None = typer.Option(
        None, "--max-parallel", help="override the YAML concurrency limit"
    ),
) -> None:
    """Fork a recorded run across variants and report Tier 1/Tier 2 checks."""
    exit_code = _run_variants(
        run_id, variants, db, upstream, port, max_parallel, require_pass=False
    )
    raise typer.Exit(code=exit_code)


@app.command()
def test(
    variants: Path = typer.Argument(..., help="variants.yaml describing variants and checks"),
    run_id: str | None = typer.Option(
        None, "--run", "-r", help="parent run id (or run_id in the YAML)"
    ),
    db: Path | None = typer.Option(None, "--db", help="cassette database path"),
    upstream: str | None = typer.Option(
        None, "--upstream", "-u", help="upstream OpenAI-compatible base URL"
    ),
    port: int = typer.Option(4040, "--port", help="daemon port (0 picks a free port)"),
    max_parallel: int | None = typer.Option(
        None, "--max-parallel", help="override the YAML concurrency limit"
    ),
) -> None:
    """Run an experiment; exit non-zero if any required check fails (CI)."""
    spec = load_experiment_spec(variants)
    parent_run_id = run_id or spec.run_id
    if not parent_run_id:
        console.print("[red]no parent run:[/red] pass --run or set run_id in the YAML")
        raise typer.Exit(code=1)
    exit_code = _run_variants(
        parent_run_id, variants, db, upstream, port, max_parallel, require_pass=True
    )
    raise typer.Exit(code=exit_code)


@app.command()
def diff(
    run_a: str = typer.Argument(..., help="first run id"),
    run_b: str = typer.Argument(..., help="second run id"),
    db: Path | None = typer.Option(None, "--db", help="cassette database path"),
) -> None:
    """Show the first step where two runs diverge."""
    db_path = resolve_db(db)
    conn = connect(db_path)
    try:
        runs = {
            run_id: conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            for run_id in (run_a, run_b)
        }
        for run_id, run in runs.items():
            if run is None:
                console.print(f"[red]unknown run:[/red] {run_id}")
                raise typer.Exit(code=1)
        steps = {
            run_id: conn.execute(
                "SELECT * FROM steps WHERE run_id = ? ORDER BY step_index", (run_id,)
            ).fetchall()
            for run_id in (run_a, run_b)
        }
        index = _first_divergent_step(steps[run_a], steps[run_b])
        if index is None:
            console.print(
                f"no divergent step: both runs have {len(steps[run_a])} "
                "identical step inputs"
            )
            return
        left = steps[run_a][index] if index < len(steps[run_a]) else None
        right = steps[run_b][index] if index < len(steps[run_b]) else None
        console.print(f"[bold]first divergence at step {index}[/bold]")
        console.print(f"  {run_a}: {_step_label(left)}")
        console.print(f"  {run_b}: {_step_label(right)}")
        blobs = BlobStore(conn)
        for run_id, row in ((run_a, left), (run_b, right)):
            if row is None:
                continue
            try:
                payload = blobs.get_json(row["input_hash"])
            except KeyError:
                payload = None
            console.print(f"[dim]{run_id} input:[/dim]")
            console.print_json(data=payload)
    finally:
        conn.close()


@app.command()
def web(
    port: int = typer.Option(4040, "--port", help="daemon port (0 picks a free port)"),
    upstream: str | None = typer.Option(
        None, "--upstream", "-u", help="upstream OpenAI-compatible base URL"
    ),
    db: Path | None = typer.Option(None, "--db", help="cassette database path"),
    no_open: bool = typer.Option(False, "--no-open", help="do not open a browser"),
) -> None:
    """Start the local daemon and open the API/UI."""
    config = load_config()
    db_path = resolve_db(db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    upstream_url = resolve_upstream(upstream, config)
    try:
        daemon = run_daemon(db_path, upstream_url, config, port=port)
    except (OSError, RuntimeError, SystemExit) as exc:
        console.print(f"[red]failed to start daemon:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    url = daemon.url
    console.print(f"checkpoint daemon at [bold]{url}[/bold] (docs {url}/docs)")
    if not no_open:
        webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        console.print("stopping daemon")
    finally:
        daemon.stop()


def _first_divergent_step(left: list[Any], right: list[Any]) -> int | None:
    for index, (step_a, step_b) in enumerate(zip(left, right, strict=False)):
        key_a = (step_a["step_type"], step_a["name"], step_a["match_key"])
        key_b = (step_b["step_type"], step_b["name"], step_b["match_key"])
        if key_a != key_b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _step_label(row: Any) -> str:
    if row is None:
        return "<no recorded step>"
    return (
        f"{row['step_type']} {row['name']!r} "
        f"(occurrence {row['occurrence']}, source {row['source']})"
    )


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


if __name__ == "__main__":
    main()
