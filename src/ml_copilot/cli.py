"""Command-line interface for ``ml-copilot``.

Subcommands are added incrementally as the project advances through
its phases. See ARCHITECTURE.md §7 for the full CLI design.

Currently implemented:
    init    — create a new ``.mlcopilot/`` project (Faz 1)
    advise  — analyze dataset + recommend models / features / pitfalls (Faz 1)

Planned:
    audit, watch, compare, evaluate, deploy, status
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
from .tools.dataset import analyze_dataset
from .ui.advise import render_analysis, render_recommendation


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
    help="ml-copilot — your AI ML engineer at every pipeline stage.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, prog_name="ml-copilot")
def cli() -> None:
    """Root command group."""


@cli.command(help="Initialize a new ml-copilot project.")
@click.argument("name")
@click.option(
    "--path",
    "parent_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Directory in which .mlcopilot/ will be created.",
)
@click.option(
    "--default-model",
    default="claude-opus-4-7",
    show_default=True,
    help="Default LLM model for this project.",
)
def init(name: str, parent_dir: Path, default_model: str) -> None:
    """Create a new ml-copilot project under ``parent_dir/.mlcopilot/``."""
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
            f"Next: [yellow]ml-copilot advise <data.csv>[/yellow]",
            title="ml-copilot init",
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
            "[dim](no .mlcopilot/ project found in current path — "
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
    """Append a single advice entry to ``.mlcopilot/advice.log``."""
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
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main() -> None:
    """Entry point for the ``ml-copilot`` console script."""
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
