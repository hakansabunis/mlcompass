"""Rich rendering for the ``audit`` command."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def render_audit(console: Console, audit: dict[str, Any]) -> None:
    """Render the result of :func:`tools.script.audit_script`."""
    console.print(_summary_panel(audit))

    findings = audit.get("findings", [])
    if not findings:
        console.print()
        console.print(
            "[green]✓ No issues detected by the static checks.[/green]"
        )
        return

    console.print()
    console.print(_findings_table(findings))
    console.print()
    _print_counts(console, findings)


# --------------------------------------------------------------------------- #
# Internal renderers                                                          #
# --------------------------------------------------------------------------- #


def _summary_panel(audit: dict[str, Any]) -> Panel:
    frameworks = audit.get("frameworks") or []
    fw_text = ", ".join(frameworks) if frameworks else "[dim]none detected[/dim]"
    body = (
        f"Path:        [cyan]{audit['path']}[/cyan]\n"
        f"Lines:       {audit['lines']}\n"
        f"Frameworks:  {fw_text}"
    )
    return Panel.fit(
        body,
        title="🔎 Script audit",
        border_style="cyan",
    )


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


def _findings_table(findings: list[dict[str, Any]]) -> Table:
    table = Table(
        title="Findings",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("Severity", style="bold")
    table.add_column("Rule")
    table.add_column("Line", justify="right")
    table.add_column("Message")

    for f in findings:
        severity = f.get("severity", "info")
        color = _SEVERITY_STYLE.get(severity, "white")
        glyph = _SEVERITY_GLYPH.get(severity, "•")
        line = "" if f.get("line") is None else str(f["line"])
        table.add_row(
            f"[{color}]{glyph} {severity}[/{color}]",
            f.get("rule_id", "—"),
            line,
            f.get("message", ""),
        )

    return table


def _print_counts(console: Console, findings: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.get("severity", "info")] = counts.get(f.get("severity", "info"), 0) + 1

    parts: list[str] = []
    for severity in ("error", "warning", "info"):
        if severity in counts:
            color = _SEVERITY_STYLE[severity]
            parts.append(f"[{color}]{counts[severity]} {severity}[/{color}]")
    summary = "   ".join(parts) if parts else "[green]all clear[/green]"
    console.print(f"Summary: {summary}")

    # Suggestions: render the first 3 suggestion lines as a hint footer.
    actionable = [f for f in findings if f.get("suggestion")][:3]
    if actionable:
        console.print()
        console.print("[cyan]💡 Suggested fixes[/cyan]")
        for f in actionable:
            console.print(f"  • [bold]{f['rule_id']}[/bold]: {f['suggestion']}")
