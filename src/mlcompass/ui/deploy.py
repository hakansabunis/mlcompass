"""Rich rendering for the ``deploy`` command."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


_STATUS_GLYPH = {"ok": "✓", "warn": "⚠", "info": "•", "fail": "✗"}
_STATUS_STYLE = {"ok": "green", "warn": "yellow", "info": "cyan", "fail": "red"}


def render_deployment(console: Console, report: dict[str, Any]) -> None:
    """Render the output of :func:`tools.deploy.assess_deployment`."""
    console.print(_summary_panel(report))

    deps = report.get("dependencies")
    if deps:
        console.print()
        console.print(_dependencies_table(deps))

    checklist = report.get("checklist") or []
    if checklist:
        console.print()
        console.print(_checklist_table(checklist))

    suggestions = (report.get("model") or {}).get("suggestions") or []
    if suggestions:
        console.print()
        console.print("[cyan]💡 Suggestions[/cyan]")
        for s in suggestions:
            console.print(f"  • {s}")

    warnings = report.get("warnings") or []
    if warnings:
        console.print()
        console.print("[yellow]⚠ Warnings[/yellow]")
        for w in warnings:
            console.print(f"  • {w}")
    else:
        console.print()
        console.print("[green]✓ No blocking warnings.[/green]")


# --------------------------------------------------------------------------- #
# Internal renderers                                                          #
# --------------------------------------------------------------------------- #


def render_deployment_advice(console: Console, advice: dict[str, Any]) -> None:
    """Render the deploy advisor (LLM layer) output."""
    verdict = advice.get("verdict")
    if verdict:
        console.print()
        console.print(Panel.fit(verdict, title="🧠 Production verdict", border_style="green"))

    blockers = advice.get("blockers") or []
    if blockers:
        console.print()
        console.print("[red]✗ Blockers[/red]")
        for b in blockers:
            console.print(f"  • {b}")

    next_steps = advice.get("next_steps") or []
    if next_steps:
        console.print()
        console.print("[cyan]🚀 Next steps[/cyan]")
        for s in next_steps:
            console.print(f"  • {s}")

    rollout = advice.get("rollout_strategy")
    if rollout:
        console.print()
        console.print(
            Panel.fit(
                f"[cyan]Rollout[/cyan]: {rollout}",
                border_style="cyan",
            )
        )


def _summary_panel(report: dict[str, Any]) -> Panel:
    model = report.get("model", {})
    body = (
        f"Model:    [cyan]{model.get('path', '—')}[/cyan]\n"
        f"Format:   [bold]{model.get('format', '—')}[/bold]\n"
        f"Size:     {model.get('size_pretty', '—')} ({model.get('size_class', '—')})\n"
        f"Target:   [bold]{report.get('target', '—')}[/bold]"
    )
    return Panel.fit(body, title="🚀 Deployment readiness", border_style="cyan")


def _dependencies_table(deps: dict[str, Any]) -> Table:
    table = Table(
        title="Dependencies",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Manifest", str(deps.get("manifest", "—")))
    table.add_row("Total", str(deps.get("count", 0)))
    pinned = len(deps.get("pinned") or [])
    unpinned = len(deps.get("unpinned") or [])
    table.add_row(
        "Pinned / unpinned",
        f"[green]{pinned}[/green] / "
        + (f"[red]{unpinned}[/red]" if unpinned else "[green]0[/green]"),
    )
    ml = deps.get("ml_packages") or []
    table.add_row(
        "ML packages",
        ", ".join(ml) if ml else "[red]none detected[/red]",
    )
    return table


def _checklist_table(checklist: list[dict[str, Any]]) -> Table:
    table = Table(
        title="Production checklist",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("", justify="center", style="bold")
    table.add_column("Item")
    table.add_column("Detail")

    for row in checklist:
        status = row.get("status", "info")
        glyph = _STATUS_GLYPH.get(status, "•")
        color = _STATUS_STYLE.get(status, "white")
        table.add_row(
            f"[{color}]{glyph}[/{color}]",
            row.get("item", "—"),
            row.get("detail", ""),
        )
    return table
