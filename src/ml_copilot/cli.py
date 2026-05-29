"""Command-line interface for ``ml-copilot``.

Subcommands are added incrementally as the project advances through
its phases. See ARCHITECTURE.md §7 for the full CLI design.

Currently implemented:
    init    — create a new ``.mlcopilot/`` project (Faz 1)

Planned:
    advise, audit, watch, compare, evaluate, deploy, status
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from . import __version__
from .context import ProjectContext, ProjectExistsError


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


def main() -> None:
    """Entry point for the ``ml-copilot`` console script."""
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
