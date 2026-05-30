"""Rich rendering + confirm prompt for permission-gated config edits."""

from __future__ import annotations

from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..tools.config_edit import ApplyResult, ConfigEdit


def make_console_confirm(console: Console, *, auto_yes: bool = False):
    """Build a confirm callback bound to ``console``.

    Args:
        console: The rich Console to render the prompt panel to.
        auto_yes: When ``True``, returns ``True`` without prompting
            (the ``--yes`` escape hatch).
    """

    def confirm(edit: ConfigEdit, _config: dict[str, Any]) -> bool:
        console.print()
        console.print(
            Panel.fit(
                _format_edit(edit),
                title="⚙ Proposed config edit",
                border_style="yellow",
            )
        )
        if auto_yes:
            console.print("[dim]--yes set; applying without prompt.[/dim]")
            return True
        return click.confirm("Apply this change?", default=False)

    return confirm


def render_apply_summary(console: Console, result: ApplyResult) -> None:
    """Print a final accounting after ``apply_edits`` returns."""
    console.print()

    if result.backup_path is not None:
        console.print(
            f"[dim]Backup written to {result.backup_path}[/dim]"
        )

    table = Table(
        title="Edit summary",
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("Status", style="bold")
    table.add_column("Key")
    table.add_column("From → To")

    for edit in result.applied:
        table.add_row(
            "[green]applied[/green]",
            edit.key,
            f"{edit.current_value!r} → {edit.proposed_value!r}",
        )
    for edit in result.rejected:
        table.add_row(
            "[red]rejected[/red]",
            edit.key,
            f"{edit.current_value!r} → {edit.proposed_value!r}",
        )
    for edit in result.skipped:
        table.add_row(
            "[dim]skipped[/dim]",
            edit.key,
            "key missing or value already matches",
        )

    if not (result.applied or result.rejected or result.skipped):
        console.print("[dim]No edits to summarise.[/dim]")
        return

    console.print(table)


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #


def _format_edit(edit: ConfigEdit) -> str:
    body = (
        f"Key:      [bold]{edit.key}[/bold]\n"
        f"Current:  {edit.current_value!r}\n"
        f"Proposed: [green]{edit.proposed_value!r}[/green]"
    )
    if edit.rationale:
        body += f"\n\n[dim]{edit.rationale}[/dim]"
    return body
