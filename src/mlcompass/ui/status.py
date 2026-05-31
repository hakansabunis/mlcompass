"""Rich rendering for the ``status`` command."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..context import ProjectContext

# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def render_status(
    console: Console,
    project: ProjectContext,
    *,
    recent_decisions: int = 5,
) -> None:
    """Print a project-context summary."""
    console.print(_project_panel(project))
    console.print()
    console.print(_state_panel(project))

    console.print()
    console.print(_command_counts_table(project.path))

    decisions = project.read_context().get("decisions") or []
    if decisions:
        console.print()
        console.print(_decisions_table(decisions[-recent_decisions:]))
    else:
        console.print()
        console.print("[dim]No decisions recorded yet.[/dim]")


# --------------------------------------------------------------------------- #
# Internal renderers                                                          #
# --------------------------------------------------------------------------- #


def _project_panel(project: ProjectContext) -> Panel:
    meta = project.project_meta
    # Look up the version under the current key (mlcompass_version)
    # first, then fall back through the two legacy spellings so older
    # projects that pre-date the v0.4 rename still render correctly.
    version = (
        meta.get("mlcompass_version")
        or meta.get("ml_compass_version")
        or meta.get("ml_copilot_version")
        or "—"
    )
    body = (
        f"Name:           [bold]{meta.get('name', '—')}[/bold]\n"
        f"Created:        {meta.get('created', '—')}\n"
        f"mlcompass ver:  {version}\n"
        f"Default model:  {meta.get('default_model', '—')}"
    )
    return Panel.fit(body, title="📁 Project", border_style="cyan")


def _state_panel(project: ProjectContext) -> Panel:
    state = project.read_context()
    active_dataset = state.get("active_dataset") or "[dim]—[/dim]"
    project_type = state.get("project_type") or "[dim]not inferred yet[/dim]"
    target = state.get("target_column") or "[dim]not detected yet[/dim]"
    current_run = state.get("current_run") or "[dim]—[/dim]"
    preferred = state.get("preferred_models") or []
    preferred_text = ", ".join(preferred) if preferred else "[dim]—[/dim]"

    body = (
        f"Active dataset:    [cyan]{active_dataset}[/cyan]\n"
        f"Project type:      {project_type}\n"
        f"Target column:     {target}\n"
        f"Current run:       {current_run}\n"
        f"Preferred models:  {preferred_text}"
    )
    return Panel.fit(body, title="🧭 Active state", border_style="magenta")


def _decisions_table(decisions: list[dict[str, Any]]) -> Table:
    table = Table(
        title=f"Recent decisions ({len(decisions)})",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("When", justify="right")
    table.add_column("Command", style="bold")
    table.add_column("Summary")

    for entry in decisions:
        timestamp = entry.get("timestamp", "")
        when = timestamp.split("T")[0] if timestamp else "—"
        table.add_row(
            when,
            entry.get("command", "—"),
            entry.get("summary", ""),
        )
    return table


def _command_counts_table(project_path: Path) -> Table:
    """Tally per-command invocations from advice.log."""
    counts: Counter[str] = Counter()
    log_path = project_path / "advice.log"
    if log_path.is_file():
        for raw in log_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            command = entry.get("command")
            if isinstance(command, str):
                counts[command] += 1

    table = Table(
        title="Command activity",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("Command", style="bold")
    table.add_column("Invocations", justify="right")

    if not counts:
        table.add_row("[dim]—[/dim]", "[dim]nothing logged yet[/dim]")
        return table

    for command, n in counts.most_common():
        table.add_row(command, str(n))
    return table
