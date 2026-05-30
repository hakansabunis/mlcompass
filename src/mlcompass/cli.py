"""Command-line interface for ``mlcompass``.

Subcommands are added incrementally as the project advances through
its phases. See ARCHITECTURE.md §7 for the full CLI design.

Currently implemented:
    init     — create a new ``.mlcompass/`` project (Faz 1)
    advise   — analyze dataset + recommend models / features / pitfalls (Faz 1)
    audit    — static analysis of a training script (Faz 2a)
    watch    — monitor a training log for anomalies (Faz 2b)
    compare  — diff two training runs side-by-side (Faz 2c)

Planned:
    evaluate, deploy, status
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import click
from rich.console import Console
from rich.panel import Panel

from . import __version__
from .agents.advise import AdvisorParseError, get_recommendation
from .context import ProjectContext, ProjectExistsError, ProjectNotFoundError
from .tools.anomaly import run_all_detectors
from .tools.dataset import analyze_dataset
from .tools.logs import merge_consecutive_same_epoch, parse_log_file
from .tools.runs import RunNotFoundError, compare_runs, load_run
from .tools.script import audit_script
from .ui.advise import render_analysis, render_recommendation
from .ui.audit import render_audit
from .ui.compare import render_compare
from .ui.watch import render_new_findings, render_watch_report


def _force_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8.

    On Windows the default encoding follows the system code page
    (e.g. ``cp1254`` for Turkish locales) and cannot represent the
    Unicode glyphs rich uses for panels, checkmarks, and box drawing.
    Reconfiguring early avoids ``UnicodeEncodeError`` at print time.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


_force_utf8_stdio()
console = Console()


@click.group(
    help="mlcompass — your AI ML engineer at every pipeline stage.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, prog_name="mlcompass")
def cli() -> None:
    """Root command group."""


@cli.command(help="Initialize a new mlcompass project.")
@click.argument("name")
@click.option(
    "--path",
    "parent_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Directory in which .mlcompass/ will be created.",
)
@click.option(
    "--default-model",
    default="claude-opus-4-7",
    show_default=True,
    help="Default LLM model for this project.",
)
def init(name: str, parent_dir: Path, default_model: str) -> None:
    """Create a new mlcompass project under ``parent_dir/.mlcompass/``."""
    try:
        ctx = ProjectContext.init(
            name=name,
            parent_dir=parent_dir,
            default_model=default_model,
        )
    except ProjectExistsError as e:
        console.print(f"[red]✗[/red] {e}")
        raise SystemExit(1) from e

    console.print(
        Panel.fit(
            f"[green]✓[/green] Project [bold]{name}[/bold] initialized.\n\n"
            f"Directory:  [cyan]{ctx.path}[/cyan]\n"
            f"Model:      {default_model}\n\n"
            f"Next: [yellow]mlcompass advise <data.csv>[/yellow]",
            title="mlcompass init",
            border_style="green",
        )
    )


@cli.command(
    help="Analyze a dataset and recommend models, features, and pitfalls."
)
@click.argument(
    "dataset_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--target",
    "target_column",
    default=None,
    help="Target column name (auto-detected from name conventions if omitted).",
)
@click.option(
    "--sample-rows",
    type=int,
    default=None,
    help="Limit analysis to the first N rows (useful for very large files).",
)
@click.option(
    "--no-llm",
    is_flag=True,
    help="Skip the LLM advisor step; show only the deterministic analysis.",
)
@click.option(
    "--model",
    "advisor_model",
    default="claude-opus-4-7",
    show_default=True,
    help="Claude model used by the advisor.",
)
def advise(
    dataset_path: Path,
    target_column: str | None,
    sample_rows: int | None,
    no_llm: bool,
    advisor_model: str,
) -> None:
    """Run the deterministic dataset analyzer, then the LLM advisor."""
    project = _try_load_project()

    with console.status("[cyan]Analyzing dataset...[/cyan]", spinner="dots"):
        analysis = analyze_dataset(
            dataset_path,
            target_column=target_column,
            sample_rows=sample_rows,
        )

    render_analysis(console, analysis)

    recommendation: dict[str, Any] | None = None

    if no_llm:
        console.print("\n[dim](--no-llm specified; skipping advisor)[/dim]")
    elif not _has_api_key():
        console.print(
            "\n[yellow]⚠ ANTHROPIC_API_KEY not set; "
            "skipping advisor step. Set the variable to enable.[/yellow]"
        )
    else:
        recommendation = _run_advisor(analysis, model=advisor_model)
        if recommendation is not None:
            render_recommendation(console, recommendation)

    if project is not None:
        _persist_advise_result(
            project=project,
            dataset_path=dataset_path,
            analysis=analysis,
            recommendation=recommendation,
        )


# --------------------------------------------------------------------------- #
# advise helpers (split out so tests can monkeypatch them)                    #
# --------------------------------------------------------------------------- #


def _try_load_project() -> ProjectContext | None:
    """Load an existing project context, or warn and return None."""
    try:
        return ProjectContext.load()
    except ProjectNotFoundError:
        console.print(
            "[dim](no .mlcompass/ project found in current path — "
            "running standalone; results will not be persisted)[/dim]\n"
        )
        return None


def _has_api_key() -> bool:
    """True iff an Anthropic API key is configured in the environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# Indirection point: tests monkeypatch ``cli._advisor_callable`` to inject a
# fake instead of calling the real ``get_recommendation``.
_advisor_callable: Callable[..., dict[str, Any]] = get_recommendation


def _run_advisor(
    analysis: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any] | None:
    """Invoke the LLM advisor, handling parse errors gracefully."""
    try:
        with console.status(
            "[cyan]Consulting model advisor...[/cyan]",
            spinner="dots",
        ):
            return _advisor_callable(analysis, model=model)
    except AdvisorParseError as exc:
        console.print(f"\n[red]✗ Advisor returned an invalid response:[/red] {exc}")
        return None


def _persist_advise_result(
    *,
    project: ProjectContext,
    dataset_path: Path,
    analysis: dict[str, Any],
    recommendation: dict[str, Any] | None,
) -> None:
    """Save dataset metadata + context updates + advice log entry."""
    fingerprint = project.register_dataset(
        dataset_path,
        meta={"analysis": analysis},
    )

    project.write_context(
        {
            "active_dataset": f"datasets/{fingerprint}.json",
            "project_type": analysis["task_hint"].get("type"),
            "target_column": analysis["target_hint"].get("column"),
        }
    )

    top_model: str
    if recommendation and recommendation.get("models"):
        top_model = recommendation["models"][0].get("name", "n/a")
    else:
        top_model = "n/a (LLM advisor skipped)"

    project.append_decision(
        command="advise",
        summary=f"Top model recommendation: {top_model}",
        reasoning=(
            f"Dataset: {dataset_path}, "
            f"task: {analysis['task_hint'].get('type', 'unknown')}"
        ),
    )

    _append_advice_log(project, dataset_path, analysis, recommendation)

    console.print(f"\n[green]✓[/green] Saved to {project.path}")


def _append_advice_log(
    project: ProjectContext,
    dataset_path: Path,
    analysis: dict[str, Any],
    recommendation: dict[str, Any] | None,
) -> None:
    """Append a single advice entry to ``.mlcompass/advice.log``."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "task_type": analysis["task_hint"].get("type"),
        "target": analysis["target_hint"].get("column"),
        "recommendation": recommendation,
    }
    log_path = project.path / "advice.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------- #
# audit command                                                               #
# --------------------------------------------------------------------------- #


@cli.command(
    help="Statically analyze a training script for common ML mistakes."
)
@click.argument(
    "script_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--skip",
    "skip_rules",
    multiple=True,
    type=click.Choice(
        [
            "seed",
            "val_split",
            "optimizer",
            "loss_stability",
            "dataloader",
            "grad_clipping",
            "eval_mode",
            "batch_size",
        ]
    ),
    help="Rule IDs to skip (may be repeated).",
)
def audit(script_path: Path, skip_rules: tuple[str, ...]) -> None:
    """Run the static auditor on ``script_path``."""
    project = _try_load_project()

    with console.status("[cyan]Auditing script...[/cyan]", spinner="dots"):
        result = audit_script(script_path, skip_rules=set(skip_rules))

    render_audit(console, result)

    if project is not None:
        _persist_audit_result(project=project, script_path=script_path, result=result)


def _persist_audit_result(
    *,
    project: ProjectContext,
    script_path: Path,
    result: dict[str, Any],
) -> None:
    """Record an audit run in the project context and advice log."""
    counts: dict[str, int] = {}
    for f in result.get("findings", []):
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    summary_parts = [f"{n} {sev}" for sev, n in counts.items()] or ["no issues"]
    summary = "Audit: " + ", ".join(summary_parts)

    project.append_decision(
        command="audit",
        summary=summary,
        reasoning=(
            f"Script: {script_path}, frameworks: "
            f"{', '.join(result.get('frameworks') or []) or 'none'}"
        ),
    )

    log_path = project.path / "advice.log"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": "audit",
        "script": str(script_path),
        "frameworks": result.get("frameworks"),
        "findings": result.get("findings"),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    console.print(f"\n[green]✓[/green] Saved to {project.path}")


# --------------------------------------------------------------------------- #
# watch command                                                               #
# --------------------------------------------------------------------------- #


@cli.command(
    help="Monitor a training log file for plateau / overfit / NaN / divergence."
)
@click.argument(
    "log_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-f",
    "--follow",
    is_flag=True,
    help="Tail the file and report new findings as they appear (Ctrl+C to stop).",
)
@click.option(
    "--poll-interval",
    type=float,
    default=1.0,
    show_default=True,
    help="Seconds between polls when --follow is active.",
)
def watch(log_path: Path, follow: bool, poll_interval: float) -> None:
    """Run the watch rules over ``log_path``."""
    project = _try_load_project()

    snapshots = merge_consecutive_same_epoch(parse_log_file(log_path))
    findings = [f.to_dict() for f in run_all_detectors(snapshots)]

    render_watch_report(
        console,
        log_path=str(log_path),
        snapshots=snapshots,
        findings=findings,
    )

    if project is not None:
        _persist_watch_result(
            project=project,
            log_path=log_path,
            snapshots=snapshots,
            findings=findings,
        )

    if follow:
        try:
            _follow_loop(
                log_path=log_path,
                seen_signatures={_finding_signature(f) for f in findings},
                snapshots=snapshots,
                poll_interval=poll_interval,
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Stopped watching.[/dim]")


def _follow_loop(
    *,
    log_path: Path,
    seen_signatures: set[str],
    snapshots: list[Any],
    poll_interval: float,
) -> None:
    """Continuously poll ``log_path`` and surface new findings."""
    import time

    last_size = log_path.stat().st_size
    console.print(f"\n[dim]Watching {log_path}... (Ctrl+C to stop)[/dim]")

    while True:
        time.sleep(poll_interval)

        size = log_path.stat().st_size
        if size <= last_size:
            continue

        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(last_size)
            new_text = f.read()
        last_size = size

        new_snapshots = merge_consecutive_same_epoch(
            snapshots + _parse_text_snapshots(new_text)
        )
        snapshots[:] = new_snapshots

        new_findings = [
            f.to_dict()
            for f in run_all_detectors(snapshots)
            if _finding_signature(f.to_dict()) not in seen_signatures
        ]
        for finding in new_findings:
            seen_signatures.add(_finding_signature(finding))

        if new_findings:
            render_new_findings(console, new_findings)


def _parse_text_snapshots(text: str) -> list[Any]:
    from .tools.logs import parse_log_text

    return parse_log_text(text)


def _finding_signature(finding: dict[str, Any]) -> str:
    """Stable identifier so we don't re-report the same finding."""
    return f"{finding['rule_id']}::{finding.get('epoch')}::{finding['message']}"


def _persist_watch_result(
    *,
    project: ProjectContext,
    log_path: Path,
    snapshots: list[Any],
    findings: list[dict[str, Any]],
) -> None:
    """Record a watch invocation in the project context + advice log."""
    counts: dict[str, int] = {}
    for f in findings:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    summary_parts = [f"{n} {sev}" for sev, n in counts.items()] or ["no issues"]
    summary = f"Watch ({log_path.name}): " + ", ".join(summary_parts)

    project.append_decision(
        command="watch",
        summary=summary,
        reasoning=f"Parsed {len(snapshots)} snapshots from {log_path}",
    )

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": "watch",
        "log": str(log_path),
        "snapshots": len(snapshots),
        "findings": findings,
    }
    with (project.path / "advice.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


# --------------------------------------------------------------------------- #
# compare command                                                             #
# --------------------------------------------------------------------------- #


@cli.command(
    help="Compare two training runs side-by-side (config + final metrics).",
)
@click.argument("run_a")
@click.argument("run_b")
def compare(run_a: str, run_b: str) -> None:
    """Compare two runs by identifier or directory path."""
    project = _try_load_project()

    try:
        record_a = load_run(run_a, project=project)
        record_b = load_run(run_b, project=project)
    except RunNotFoundError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise SystemExit(1) from exc

    comparison = compare_runs(record_a, record_b)
    render_compare(console, comparison)

    if project is not None:
        _persist_compare_result(
            project=project,
            run_a=record_a.id,
            run_b=record_b.id,
            comparison=comparison,
        )


def _persist_compare_result(
    *,
    project: ProjectContext,
    run_a: str,
    run_b: str,
    comparison: dict[str, Any],
) -> None:
    """Record a compare invocation in the project's decision log."""
    verdict = comparison.get("verdict", "inconclusive")
    summary = f"Compare {run_a} vs {run_b}: {verdict}"
    project.append_decision(
        command="compare",
        summary=summary,
        reasoning=comparison.get("verdict_explanation", ""),
    )

    log_path = project.path / "advice.log"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": "compare",
        "run_a": run_a,
        "run_b": run_b,
        "verdict": verdict,
        "verdict_explanation": comparison.get("verdict_explanation"),
        "config_diff": comparison.get("config_diff"),
        "metric_comparison": comparison.get("metric_comparison"),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main() -> None:
    """Entry point for the ``mlcompass`` console script."""
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
