"""Rich rendering for the ``evaluate`` command."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def render_evaluation(console: Console, result: dict[str, Any]) -> None:
    """Render the deterministic output of :func:`tools.evaluation.evaluate`."""
    console.print(_summary_panel(result))

    metrics = result.get("metrics", {})
    if metrics:
        console.print()
        console.print(_metrics_table(metrics, result.get("task")))

    task = result.get("task")
    if task == "binary_classification":
        _render_binary_extras(console, result)
    elif task == "multiclass_classification":
        _render_multiclass_extras(console, result)
    elif task == "regression":
        _render_regression_extras(console, result)

    hard = result.get("hard_examples") or []
    if hard:
        console.print()
        console.print(_hard_examples_table(hard, task))

    warnings = result.get("warnings") or []
    if warnings:
        console.print()
        console.print("[yellow]⚠ Warnings[/yellow]")
        for w in warnings:
            console.print(f"  • {w}")
    else:
        console.print()
        console.print("[green]✓ No evaluation warnings.[/green]")


# --------------------------------------------------------------------------- #
# Sections                                                                    #
# --------------------------------------------------------------------------- #


def _summary_panel(result: dict[str, Any]) -> Panel:
    cols = result.get("columns") or {}
    body = (
        f"Task:    [bold]{_format_task(result.get('task'))}[/bold]\n"
        f"Rows:    {result.get('rows', 0):,}\n"
        f"y_true:  [cyan]{cols.get('y_true', '—')}[/cyan]\n"
        f"y_pred:  [cyan]{cols.get('y_pred') or '—'}[/cyan]"
    )
    if cols.get("y_prob"):
        body += f"\ny_prob:  [cyan]{cols['y_prob']}[/cyan]"
    return Panel.fit(body, title="📊 Evaluation", border_style="cyan")


def _metrics_table(metrics: dict[str, Any], task: str | None) -> Table:
    table = Table(
        title="Metrics",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for name, value in metrics.items():
        table.add_row(name, _format_number(value))
    return table


def _render_binary_extras(console: Console, result: dict[str, Any]) -> None:
    cm = result.get("confusion_matrix")
    if cm:
        console.print()
        console.print(_binary_confusion_panel(cm, result.get("positive_label")))

    sweep = result.get("threshold_sweep") or []
    if sweep:
        console.print()
        console.print(_threshold_sweep_table(sweep, result.get("best_threshold")))


def _binary_confusion_panel(cm: dict[str, int], positive_label: Any) -> Panel:
    body = (
        "                  Predicted\n"
        f"                 {positive_label!s:>6}    not\n"
        f" Actual {positive_label!s:>6}  {cm['tp']:>6}  {cm['fn']:>6}\n"
        f"        not     {cm['fp']:>6}  {cm['tn']:>6}"
    )
    return Panel.fit(
        f"[white]{body}[/white]",
        title="Confusion matrix",
        border_style="magenta",
    )


def _threshold_sweep_table(
    sweep: list[dict[str, float]],
    best: dict[str, float] | None,
) -> Table:
    table = Table(
        title="Threshold sweep",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("Threshold", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    best_thr = best["threshold"] if best else None
    for row in sweep:
        is_best = (best_thr is not None) and row["threshold"] == best_thr
        thr_cell = _format_number(row["threshold"])
        prec_cell = _format_number(row["precision"])
        rec_cell = _format_number(row["recall"])
        f1_cell = _format_number(row["f1"])
        if is_best:
            thr_cell = f"[green]{thr_cell}[/green]"
            f1_cell = f"[green]{f1_cell} ★[/green]"
        table.add_row(thr_cell, prec_cell, rec_cell, f1_cell)
    return table


def _render_multiclass_extras(console: Console, result: dict[str, Any]) -> None:
    per_class = result.get("per_class") or []
    if per_class:
        console.print()
        console.print(_per_class_table(per_class))

    cm = result.get("confusion_matrix")
    labels = result.get("labels") or []
    if cm and labels and len(labels) <= 8:
        console.print()
        console.print(_multiclass_confusion_table(cm, labels))


def _per_class_table(per_class: list[dict[str, Any]]) -> Table:
    table = Table(
        title="Per-class metrics",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("Label", style="bold")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Support", justify="right")
    for row in per_class:
        table.add_row(
            str(row.get("label", "—")),
            _format_number(row.get("precision", 0)),
            _format_number(row.get("recall", 0)),
            _format_number(row.get("f1", 0)),
            str(row.get("support", 0)),
        )
    return table


def _multiclass_confusion_table(
    cm: list[list[int]],
    labels: list[Any],
) -> Table:
    table = Table(
        title="Confusion matrix",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("True \\ Pred", style="bold")
    for label in labels:
        table.add_column(str(label), justify="right")
    for i, row in enumerate(cm):
        cells = [str(labels[i])] + [str(int(v)) for v in row]
        table.add_row(*cells)
    return table


def _render_regression_extras(console: Console, result: dict[str, Any]) -> None:
    residuals = result.get("residuals")
    if residuals:
        body = (
            f"Mean:   {_format_number(residuals.get('mean', 0))}\n"
            f"Std:    {_format_number(residuals.get('std', 0))}\n"
            f"Min:    {_format_number(residuals.get('min', 0))}\n"
            f"Max:    {_format_number(residuals.get('max', 0))}\n"
            f"Count:  {residuals.get('count', 0):,}"
        )
        console.print()
        console.print(
            Panel.fit(body, title="Residuals", border_style="magenta")
        )


def _hard_examples_table(
    hard: list[dict[str, Any]],
    task: str | None,
) -> Table:
    table = Table(
        title="Hard examples (top-k worst)",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("#", justify="right")
    table.add_column("True")
    table.add_column("Pred")
    if task == "binary_classification":
        table.add_column("Prob", justify="right")
    elif task == "regression":
        table.add_column("Residual", justify="right")
    for example in hard:
        cells = [
            str(example.get("row_index", "—")),
            str(example.get("true_label", example.get("y_true", "—"))),
            str(example.get("predicted_label", example.get("y_pred", "—"))),
        ]
        if task == "binary_classification":
            prob = example.get("probability")
            cells.append("—" if prob is None else _format_number(prob))
        elif task == "regression":
            cells.append(_format_number(example.get("residual", 0)))
        table.add_row(*cells)
    return table


# --------------------------------------------------------------------------- #
# Formatting                                                                  #
# --------------------------------------------------------------------------- #


def _format_task(value: str | None) -> str:
    if not value:
        return "unknown"
    return value.replace("_", " ")


def _format_number(value: float | int) -> str:
    if value == 0:
        return "0"
    if not isinstance(value, (int, float)):
        return str(value)
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.2f}"
    if magnitude >= 1:
        return f"{value:.4f}".rstrip("0").rstrip(".") or "0"
    if magnitude >= 0.0001:
        return f"{value:.4f}".rstrip("0").rstrip(".") or "0"
    return f"{value:.2e}"
