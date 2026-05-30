"""Rich rendering for the ``compare`` command."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def render_compare(console: Console, comparison: dict[str, Any]) -> None:
    """Render the output of :func:`tools.runs.compare_runs`."""
    console.print(_header_panel(comparison))

    metric_rows = comparison.get("metric_comparison") or []
    if metric_rows:
        console.print()
        console.print(_metrics_table(metric_rows))
    else:
        console.print()
        console.print(
            "[dim]No common metrics with a known direction to compare.[/dim]"
        )

    config_rows = comparison.get("config_diff") or []
    if config_rows:
        console.print()
        console.print(_config_table(config_rows))

    console.print()
    console.print(_verdict_panel(comparison))


# --------------------------------------------------------------------------- #
# Internal renderers                                                          #
# --------------------------------------------------------------------------- #


def _header_panel(comparison: dict[str, Any]) -> Panel:
    a = comparison["run_a"]
    b = comparison["run_b"]
    body = (
        f"[bold]Run A[/bold]  [cyan]{a['id']}[/cyan]"
        + (f"  [dim]({a['name']})[/dim]" if a.get("name") else "")
        + f"  · {a['epochs']} epoch(s)\n"
        f"[bold]Run B[/bold]  [cyan]{b['id']}[/cyan]"
        + (f"  [dim]({b['name']})[/dim]" if b.get("name") else "")
        + f"  · {b['epochs']} epoch(s)"
    )
    return Panel.fit(body, title="🆚 Run comparison", border_style="cyan")


def _metrics_table(rows: list[dict[str, Any]]) -> Table:
    table = Table(
        title="Final-epoch metrics",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Run A", justify="right")
    table.add_column("Run B", justify="right")
    table.add_column("Δ (B − A)", justify="right")
    table.add_column("Winner")

    for row in rows:
        better = row.get("better")
        winner_marker = {
            "a": "[green]A[/green]",
            "b": "[green]B[/green]",
            "tie": "[yellow]tie[/yellow]",
            None: "[dim]—[/dim]",
        }.get(better, "[dim]—[/dim]")
        delta = row["delta"]
        delta_str = _format_number(delta, signed=True)
        # Colour delta cell to match the winner so it reads at a glance.
        if better == "b":
            delta_str = f"[green]{delta_str}[/green]"
        elif better == "a":
            delta_str = f"[red]{delta_str}[/red]"

        table.add_row(
            row["name"],
            _format_number(row["a_value"]),
            _format_number(row["b_value"]),
            delta_str,
            winner_marker,
        )
    return table


def _config_table(rows: list[dict[str, Any]]) -> Table:
    table = Table(
        title="Config differences",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("Hyperparameter", style="bold")
    table.add_column("Run A")
    table.add_column("Run B")

    for row in rows:
        table.add_row(
            row["key"],
            _format_value(row["a_value"]),
            _format_value(row["b_value"]),
        )
    return table


def render_compare_hypothesis(console: Console, hypothesis: dict[str, Any]) -> None:
    """Render the compare hypothesizer (LLM layer) output."""
    text = hypothesis.get("hypothesis")
    if text:
        console.print()
        console.print(
            Panel.fit(text, title="🧠 LLM hypothesis", border_style="green")
        )

    factors = hypothesis.get("key_factors") or []
    if factors:
        table = Table(
            title="Key factors",
            show_header=True,
            header_style="bold green",
            title_justify="left",
        )
        table.add_column("Config key", style="bold")
        table.add_column("Impact")
        table.add_column("Reason")
        for factor in factors:
            impact = factor.get("impact", "—")
            impact_color = {
                "high": "green",
                "medium": "yellow",
                "low": "red",
            }.get(impact, "white")
            table.add_row(
                factor.get("config_key", "—"),
                f"[{impact_color}]{impact}[/{impact_color}]",
                factor.get("reason", "—"),
            )
        console.print()
        console.print(table)

    next_exp = hypothesis.get("next_experiment")
    if next_exp:
        console.print()
        console.print(
            Panel.fit(
                f"[cyan]Next experiment[/cyan]: {next_exp}",
                border_style="cyan",
            )
        )


def _verdict_panel(comparison: dict[str, Any]) -> Panel:
    verdict = comparison.get("verdict", "inconclusive")
    text = comparison.get("verdict_explanation", "")

    style, icon, headline = {
        "a_better": ("green", "🏆", "Run A wins"),
        "b_better": ("green", "🏆", "Run B wins"),
        "mixed": ("yellow", "⚖️ ", "Mixed result"),
        "inconclusive": ("dim", "❔", "Inconclusive"),
    }.get(verdict, ("dim", "❔", verdict))

    return Panel.fit(
        f"{icon}  [bold]{headline}[/bold]\n{text}",
        border_style=style,
    )


# --------------------------------------------------------------------------- #
# Formatting helpers                                                          #
# --------------------------------------------------------------------------- #


def _format_number(value: float, *, signed: bool = False) -> str:
    """Compact numeric formatter for the metric and delta cells."""
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 100:
        s = f"{value:,.2f}"
    elif magnitude >= 1:
        s = f"{value:.4f}".rstrip("0").rstrip(".")
        if not s:
            s = "0"
    elif magnitude >= 0.0001:
        s = f"{value:.4f}".rstrip("0").rstrip(".")
        if not s:
            s = "0"
    else:
        s = f"{value:.2e}"
    if signed and not s.startswith("-"):
        s = f"+{s}"
    return s


def _format_value(value: Any) -> str:
    """Stringify a config value compactly."""
    if value is None:
        return "[dim]—[/dim]"
    if isinstance(value, float):
        return _format_number(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    return str(value)
