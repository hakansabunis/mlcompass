"""Rich rendering for the ``watch`` command."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def render_watch_report(
    console: Console,
    *,
    log_path: str,
    snapshots: list[Any],  # list[MetricSnapshot] — duck-typed to avoid import cycle
    findings: list[dict[str, Any]],
) -> None:
    """One-shot watch report — overview panel + recent metrics + findings."""
    console.print(_overview_panel(log_path, snapshots, findings))

    if snapshots:
        console.print()
        console.print(_recent_metrics_table(snapshots))

    if findings:
        console.print()
        console.print(_findings_table(findings))
        console.print()
        _print_top_suggestions(console, findings)
    else:
        console.print()
        console.print("[green]✓ No anomalies detected by the watch rules.[/green]")


def render_watch_diagnosis(console: Console, diagnosis: dict[str, Any]) -> None:
    """Render the watch diagnostician (LLM layer) output."""
    entries = diagnosis.get("diagnosis") or []
    if entries:
        table = Table(
            title="🧠 LLM diagnosis",
            show_header=True,
            header_style="bold green",
            title_justify="left",
        )
        table.add_column("Rule", style="bold")
        table.add_column("Hypothesis")
        table.add_column("Recommended action")
        table.add_column("Conf.")
        for entry in entries:
            conf = entry.get("confidence", "—")
            conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(conf, "white")
            table.add_row(
                entry.get("finding_rule_id", "—"),
                entry.get("hypothesis", "—"),
                entry.get("recommended_action", "—"),
                f"[{conf_color}]{conf}[/{conf_color}]",
            )
        console.print()
        console.print(table)

    summary = diagnosis.get("summary")
    if summary:
        console.print()
        console.print(Panel.fit(summary, title="Summary", border_style="green"))


def render_new_findings(console: Console, findings: list[dict[str, Any]]) -> None:
    """Print only the freshly-detected findings during ``--follow`` mode."""
    if not findings:
        return
    console.print()
    console.print(_findings_table(findings))


# --------------------------------------------------------------------------- #
# Internal renderers                                                          #
# --------------------------------------------------------------------------- #


_SEVERITY_STYLE = {
    "error": "red",
    "warning": "yellow",
    "info": "cyan",
}

_SEVERITY_GLYPH = {
    "error": "✗",
    "warning": "⚠",
    "info": "ℹ",
}


def _overview_panel(
    log_path: str,
    snapshots: list[Any],
    findings: list[dict[str, Any]],
) -> Panel:
    last_epoch = "—"
    if snapshots:
        for snap in reversed(snapshots):
            if snap.epoch is not None:
                last_epoch = str(snap.epoch)
                break

    counts: dict[str, int] = {}
    for f in findings:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    counts_text = (
        "   ".join(
            f"[{_SEVERITY_STYLE[sev]}]{counts[sev]} {sev}[/{_SEVERITY_STYLE[sev]}]"
            for sev in ("error", "warning", "info")
            if sev in counts
        )
        or "[green]none[/green]"
    )

    body = (
        f"Log:        [cyan]{log_path}[/cyan]\n"
        f"Snapshots:  {len(snapshots)}\n"
        f"Last epoch: {last_epoch}\n"
        f"Findings:   {counts_text}"
    )
    return Panel.fit(body, title="👁  Watch report", border_style="cyan")


def _recent_metrics_table(snapshots: list[Any], rows: int = 8) -> Table:
    table = Table(
        title=f"Recent metrics (last {min(rows, len(snapshots))})",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )

    # Collect column names from the last N snapshots so the table reflects
    # the rolling window the user actually cares about.
    tail = snapshots[-rows:]
    metric_names: list[str] = []
    for snap in tail:
        for name in snap.metrics:
            if name not in metric_names:
                metric_names.append(name)

    table.add_column("Epoch", justify="right")
    table.add_column("Step", justify="right")
    for name in metric_names:
        table.add_column(name, justify="right")

    for snap in tail:
        row = [
            "" if snap.epoch is None else str(snap.epoch),
            "" if snap.step is None else str(snap.step),
        ]
        for name in metric_names:
            value = snap.metrics.get(name)
            row.append("" if value is None else _format_number(value))
        table.add_row(*row)
    return table


def _findings_table(findings: list[dict[str, Any]]) -> Table:
    table = Table(
        title="Findings",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("Severity", style="bold")
    table.add_column("Rule")
    table.add_column("Epoch", justify="right")
    table.add_column("Message")

    for f in findings:
        severity = f.get("severity", "info")
        color = _SEVERITY_STYLE.get(severity, "white")
        glyph = _SEVERITY_GLYPH.get(severity, "•")
        epoch = "" if f.get("epoch") is None else str(f["epoch"])
        table.add_row(
            f"[{color}]{glyph} {severity}[/{color}]",
            f.get("rule_id", "—"),
            epoch,
            f.get("message", ""),
        )
    return table


def _print_top_suggestions(console: Console, findings: list[dict[str, Any]]) -> None:
    actionable = [f for f in findings if f.get("suggestion")][:3]
    if not actionable:
        return
    console.print("[cyan]💡 Suggested fixes[/cyan]")
    for f in actionable:
        console.print(f"  • [bold]{f['rule_id']}[/bold]: {f['suggestion']}")


# --------------------------------------------------------------------------- #
# Formatting                                                                  #
# --------------------------------------------------------------------------- #


def _format_number(value: float) -> str:
    if value == 0:
        return "0"
    if value != value:  # NaN
        return "[red]NaN[/red]"
    if value == float("inf") or value == float("-inf"):
        return "[red]Inf[/red]"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.2f}"
    if magnitude >= 0.0001:
        return f"{value:.4f}".rstrip("0").rstrip(".") or "0"
    return f"{value:.2e}"
