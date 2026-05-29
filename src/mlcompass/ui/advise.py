"""Rich rendering for the ``advise`` command.

Splits naturally into two phases:

1. ``render_analysis``: the deterministic Python output from
   ``tools.dataset.analyze_dataset`` — shape, target hint, task hint,
   per-column summary, and warnings.
2. ``render_recommendation``: the LLM advisor's JSON recommendation —
   model table, feature engineering suggestions, pitfalls.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def render_analysis(console: Console, analysis: dict[str, Any]) -> None:
    """Render the deterministic dataset analysis."""
    console.print(_summary_panel(analysis))
    console.print(_columns_table(analysis["columns"]))
    if analysis.get("warnings"):
        console.print()
        console.print("[yellow]⚠ Warnings[/yellow]")
        for warning in analysis["warnings"]:
            console.print(f"  • {warning}")


def render_recommendation(console: Console, rec: dict[str, Any]) -> None:
    """Render the advisor's recommendation."""
    if rec.get("models"):
        console.print()
        console.print(_models_table(rec["models"]))

    if rec.get("features"):
        console.print()
        console.print("[cyan]🔧 Feature engineering[/cyan]")
        for feat in rec["features"]:
            col = feat.get("column", "—")
            suggestion = feat.get("suggestion", "")
            console.print(f"  • [bold]{col}[/bold] → {suggestion}")
            if feat.get("reason"):
                console.print(f"     [dim]{feat['reason']}[/dim]")

    if rec.get("pitfalls"):
        console.print()
        console.print("[red]⚠ Pitfalls[/red]")
        for pitfall in rec["pitfalls"]:
            issue = pitfall.get("issue", "—")
            mitigation = pitfall.get("mitigation", "")
            console.print(f"  • {issue}")
            if mitigation:
                console.print(f"     [dim]→ {mitigation}[/dim]")


# --------------------------------------------------------------------------- #
# Internal renderers                                                          #
# --------------------------------------------------------------------------- #


def _summary_panel(analysis: dict[str, Any]) -> Panel:
    shape = analysis["shape"]
    body = (
        f"Path:    [cyan]{analysis['path']}[/cyan]\n"
        f"Shape:   {shape['rows']:,} rows × {shape['cols']} columns\n"
        f"Target:  {_format_target(analysis['target_hint'])}\n"
        f"Task:    {_format_task(analysis['task_hint'])}"
    )
    return Panel.fit(
        body,
        title="📊 Dataset analysis",
        border_style="cyan",
    )


def _columns_table(columns: list[dict[str, Any]]) -> Table:
    table = Table(
        title="Columns",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("Name", style="bold")
    table.add_column("Type")
    table.add_column("Missing", justify="right")
    table.add_column("Notes")

    for col in columns:
        table.add_row(
            col["name"],
            col["type"],
            _format_missing(col["missing_pct"]),
            _column_notes(col),
        )
    return table


def _models_table(models: list[dict[str, Any]]) -> Table:
    table = Table(
        title="✨ Recommended models",
        show_header=True,
        header_style="bold green",
        title_justify="left",
    )
    table.add_column("Model", style="bold")
    table.add_column("Expected metric")
    table.add_column("Why")

    for model in models:
        table.add_row(
            model.get("name", "—"),
            model.get("expected_metric", "—"),
            model.get("reason", "—"),
        )
    return table


# --------------------------------------------------------------------------- #
# Formatting helpers                                                          #
# --------------------------------------------------------------------------- #


def _format_target(target_hint: dict[str, Any]) -> str:
    col = target_hint.get("column")
    conf = target_hint.get("confidence", "none")
    if col is None:
        return "[yellow]not detected[/yellow] (specify with --target)"
    color = {
        "explicit": "green",
        "high": "green",
        "medium": "yellow",
        "low": "yellow",
        "none": "red",
    }.get(conf, "white")
    return f"[bold]{col}[/bold] ([{color}]{conf} confidence[/{color}])"


def _format_task(task_hint: dict[str, Any]) -> str:
    kind = task_hint.get("type", "unknown")
    if kind == "binary_classification":
        balance = task_hint.get("class_balance", {})
        if balance:
            parts = [f"{k}={v * 100:.0f}%" for k, v in balance.items()]
            return f"binary classification ({', '.join(parts)})"
        return "binary classification"
    if kind == "multiclass_classification":
        n = task_hint.get("n_classes", "?")
        return f"multiclass classification ({n} classes)"
    if kind == "regression":
        stats = task_hint.get("target_stats", {})
        if "min" in stats and "max" in stats:
            return f"regression (range {stats['min']:.2f} → {stats['max']:.2f})"
        return "regression"
    return kind


def _format_missing(pct: float) -> str:
    if pct == 0:
        return "0%"
    if pct < 0.01:
        return f"{pct * 100:.2f}%"
    if pct < 0.5:
        return f"[yellow]{pct * 100:.1f}%[/yellow]"
    return f"[red]{pct * 100:.1f}%[/red]"


def _column_notes(col: dict[str, Any]) -> str:
    """One-line summary appropriate to the column's type."""
    col_type = col.get("type")
    if col_type == "numeric":
        stats = col.get("stats")
        outliers = col.get("outliers", {}) or {}
        if stats is None:
            return "[dim]no data[/dim]"
        n_outliers = outliers.get("iqr_count", 0)
        if n_outliers > 0:
            return (
                f"range [{stats['min']:.1f}, {stats['max']:.1f}], "
                f"[yellow]{n_outliers} IQR outliers[/yellow]"
            )
        return f"range [{stats['min']:.1f}, {stats['max']:.1f}]"
    if col_type == "categorical":
        card = col.get("cardinality", 0)
        if card > 50:
            return f"[yellow]cardinality {card}[/yellow]"
        return f"cardinality {card}"
    if col_type == "datetime":
        date_range = col.get("range")
        if date_range:
            return f"{date_range['min']} → {date_range['max']}"
        return "no data"
    if col_type == "boolean":
        return f"{col.get('true_count', 0)} true / {col.get('false_count', 0)} false"
    if col_type == "text":
        card = col.get("cardinality", 0)
        avg_len = col.get("avg_length", 0)
        return f"avg len {avg_len:.0f}, cardinality {card}"
    return ""
