"""Rich rendering for the ``monitor`` command."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Severity → colour map.
_SEVERITY_COLOUR = {
    "stable": "green",
    "moderate": "yellow",
    "major": "red",
    "skipped": "dim",
}


def render_drift(console: Console, result: dict[str, Any]) -> None:
    """Print the full drift report."""
    console.print(_summary_panel(result))
    console.print()
    console.print(_top_drifted_table(result))
    console.print()
    console.print(_per_feature_table(result))
    if result.get("warnings"):
        console.print()
        console.print(_warnings_panel(result["warnings"]))
    console.print()
    console.print(_verdict_panel(result["verdict"]))


def render_drift_interpretation(console: Console, interpretation: dict[str, Any]) -> None:
    """Render the optional ``--llm`` interpreter output."""
    headline = interpretation.get("headline", "")
    likely_cause = interpretation.get("likely_cause", "")
    next_steps = interpretation.get("next_steps") or []

    body_parts: list[str] = []
    if headline:
        body_parts.append(f"[bold]{headline}[/bold]")
    if likely_cause:
        body_parts.append(f"\n[dim]Likely cause:[/dim] {likely_cause}")
    if next_steps:
        body_parts.append("\n[bold]Next steps[/bold]")
        for i, step in enumerate(next_steps, 1):
            body_parts.append(f"  {i}. {step}")

    body = "\n".join(body_parts).strip() or "[dim](empty interpretation)[/dim]"
    console.print(Panel(body, title="🤖 Drift interpretation", border_style="magenta"))


# --------------------------------------------------------------------------- #
# Internal renderers                                                          #
# --------------------------------------------------------------------------- #


def _summary_panel(result: dict[str, Any]) -> Panel:
    agg = result["aggregate"]
    mean_psi = agg.get("mean_psi")
    max_psi = agg.get("max_psi")
    mean_str = f"{mean_psi:.4f}" if mean_psi is not None else "—"
    max_str = f"{max_psi:.4f}" if max_psi is not None else "—"
    body = (
        f"Reference rows:  [bold]{result['reference_rows']}[/bold]\n"
        f"Current rows:    [bold]{result['current_rows']}[/bold]\n"
        f"Features:        {len(result['features'])}\n"
        f"Mean PSI:        {mean_str}\n"
        f"Max  PSI:        {max_str}\n"
        f"Stable / Moderate / Major: "
        f"[green]{agg.get('n_stable', 0)}[/green] / "
        f"[yellow]{agg.get('n_moderate', 0)}[/yellow] / "
        f"[red]{agg.get('n_major', 0)}[/red]"
    )
    return Panel.fit(body, title="📊 Drift summary", border_style="cyan")


def _top_drifted_table(result: dict[str, Any]) -> Table:
    table = Table(
        title="Top drifted features",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("Feature", style="bold")
    table.add_column("PSI", justify="right")
    table.add_column("Kind")
    table.add_column("Severity", justify="right")

    top = result.get("top_drifted") or []
    if not top:
        table.add_row("—", "—", "—", "[dim]no PSI-eligible features[/dim]")
        return table
    for entry in top:
        colour = _SEVERITY_COLOUR.get(entry["severity"], "white")
        table.add_row(
            entry["feature"],
            f"{entry['psi']:.4f}",
            entry["kind"],
            f"[{colour}]{entry['severity']}[/{colour}]",
        )
    return table


def _per_feature_table(result: dict[str, Any]) -> Table:
    table = Table(
        title="Per-feature drift",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("Feature", style="bold")
    table.add_column("Kind")
    table.add_column("PSI", justify="right")
    table.add_column("KS / chi²", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Severity", justify="right")

    rows = result.get("feature_results") or []
    for row in rows:
        psi = row.get("psi")
        psi_str = f"{psi:.4f}" if psi is not None else "—"
        if row["kind"] == "numeric":
            stat = row.get("ks_stat")
            pval = row.get("ks_pvalue")
        else:
            stat = row.get("chi2_stat")
            pval = row.get("chi2_pvalue")
        stat_str = f"{stat:.4f}" if stat is not None else "—"
        pval_str = _format_p(pval) if pval is not None else "—"
        colour = _SEVERITY_COLOUR.get(row["severity"], "white")
        table.add_row(
            row["feature"],
            row["kind"],
            psi_str,
            stat_str,
            pval_str,
            f"[{colour}]{row['severity']}[/{colour}]",
        )
    return table


def _warnings_panel(warnings: list[str]) -> Panel:
    body = "\n".join(f"• {w}" for w in warnings)
    return Panel.fit(body, title="⚠ Warnings", border_style="yellow")


def _verdict_panel(verdict: dict[str, Any]) -> Panel:
    status = verdict.get("status", "unknown")
    colour_map = {
        "stable": "green",
        "moderate_drift": "yellow",
        "major_drift": "red",
        "unknown": "dim",
    }
    colour = colour_map.get(status, "white")
    retrain = (
        "[bold red]Retrain recommended[/bold red]"
        if verdict.get("retrain_recommended")
        else "[dim]No retrain needed[/dim]"
    )
    body = f"{verdict.get('message', '')}\n\n{retrain}"
    return Panel.fit(body, title=f"⚖ Verdict: {status}", border_style=colour)


def _format_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"
