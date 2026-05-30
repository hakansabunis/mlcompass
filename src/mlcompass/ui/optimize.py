"""Rich rendering for the ``optimize`` command."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def render_optimize(console: Console, result: dict[str, Any]) -> None:
    """Print the full optimize report."""
    console.print(_summary_panel(result))
    console.print()
    console.print(_leaderboard_table(result))
    console.print()
    console.print(_sensitivity_table(result))
    console.print()
    console.print(_suggestions_panel(result))
    if result.get("warnings"):
        console.print()
        console.print(_warnings_panel(result["warnings"]))


def render_optimize_strategy(console: Console, strategy: dict[str, Any]) -> None:
    """Render the optional ``--llm`` strategist output."""
    headline = strategy.get("headline", "")
    pattern = strategy.get("pattern", "")
    next_plan = strategy.get("next_plan") or []

    body_parts: list[str] = []
    if headline:
        body_parts.append(f"[bold]{headline}[/bold]")
    if pattern:
        body_parts.append(f"\n[dim]Pattern detected:[/dim] {pattern}")
    if next_plan:
        body_parts.append("\n[bold]Suggested plan[/bold]")
        for i, step in enumerate(next_plan, 1):
            body_parts.append(f"  {i}. {step}")

    body = "\n".join(body_parts).strip() or "[dim](empty strategy)[/dim]"
    console.print(Panel(body, title="🧠 HPO strategist", border_style="magenta"))


# --------------------------------------------------------------------------- #
# Internal renderers                                                          #
# --------------------------------------------------------------------------- #


def _summary_panel(result: dict[str, Any]) -> Panel:
    best = result["best"]
    body = (
        f"Metric:           [bold]{result['metric']}[/bold] "
        f"([italic]{result['direction']}[/italic])\n"
        f"Runs analysed:    {result['n_scored']} / {result['n_runs']}\n"
        f"Best score:       [bold green]{best['score']:.6g}[/bold green]\n"
        f"Best run:         [cyan]{best['id']}[/cyan]"
        + (f" ({best['name']})" if best.get("name") else "")
    )
    return Panel.fit(body, title="🎯 Optimize summary", border_style="cyan")


def _leaderboard_table(result: dict[str, Any]) -> Table:
    table = Table(
        title="Leaderboard",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("Rank", justify="right")
    table.add_column("Run", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Config (key knobs)")

    for i, entry in enumerate(result["leaderboard"], 1):
        cfg = entry.get("config") or {}
        # Show up to 4 hyperparameters compactly.
        cfg_str = ", ".join(f"{k}={_fmt(v)}" for k, v in list(cfg.items())[:4])
        table.add_row(
            str(i),
            entry["id"],
            f"{entry['score']:.6g}",
            cfg_str,
        )
    return table


def _sensitivity_table(result: dict[str, Any]) -> Table:
    table = Table(
        title="Sensitivity (rank correlation with metric)",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("Hyperparameter", style="bold")
    table.add_column("Kind")
    table.add_column("Correlation", justify="right")
    table.add_column("Observations", justify="right")

    sens = result.get("sensitivity") or []
    if not sens:
        table.add_row("—", "—", "—", "[dim]too few runs for signal[/dim]")
        return table
    for entry in sens:
        corr = entry.get("correlation")
        corr_str = f"{corr:+.3f}" if corr is not None else "—"
        colour = _corr_colour(corr)
        table.add_row(
            entry["hyperparam"],
            entry["kind"],
            f"[{colour}]{corr_str}[/{colour}]",
            str(entry.get("n_observations", "—")),
        )
    return table


def _suggestions_panel(result: dict[str, Any]) -> Panel:
    suggestions = result.get("suggestions") or []
    if not suggestions:
        return Panel.fit(
            "[dim]No suggestions available.[/dim]",
            title="🧪 Next configs to try",
            border_style="dim",
        )
    lines: list[str] = []
    for i, sug in enumerate(suggestions, 1):
        cfg = sug.get("config") or {}
        cfg_str = ", ".join(f"{k}={_fmt(v)}" for k, v in cfg.items())
        lines.append(f"[bold]{i}.[/bold] {cfg_str}")
        lines.append(f"   [dim]{sug.get('rationale', '')}[/dim]")
        lines.append("")
    return Panel.fit(
        "\n".join(lines).rstrip(), title="🧪 Next configs to try", border_style="green"
    )


def _warnings_panel(warnings: list[str]) -> Panel:
    body = "\n".join(f"• {w}" for w in warnings)
    return Panel.fit(body, title="⚠ Warnings", border_style="yellow")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return json.dumps(value, default=str)


def _corr_colour(corr: float | None) -> str:
    if corr is None:
        return "dim"
    if abs(corr) >= 0.5:
        return "bold red"
    if abs(corr) >= 0.2:
        return "yellow"
    return "white"
