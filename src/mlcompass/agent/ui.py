"""Rich streaming UI for ``mlcompass agent``.

One ``StepRenderer`` instance keeps a ``rich.console.Console`` around
and renders each ``AgentStep`` as a discrete block — coloured by event
kind so the reader can scan a long run quickly.

A second helper (``console_confirm``) makes the permission callback
the orchestrator hands the backend: shows the requested tool + its
arguments, asks ``y/N``, returns a bool.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax

from .backends import AgentStep, PermissionCallback


class StepRenderer:
    """Render each :class:`AgentStep` as a discrete block."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def __call__(self, step: AgentStep) -> None:
        if step.kind == "message":
            text = (step.content or "").strip()
            if text:
                self.console.print(
                    Panel.fit(
                        text,
                        title="[bold cyan]agent[/bold cyan]",
                        border_style="cyan",
                    )
                )
        elif step.kind == "thinking":
            text = (step.content or "").strip()
            if text:
                self.console.print(
                    Panel.fit(
                        text,
                        title="[dim]thinking[/dim]",
                        border_style="dim",
                    )
                )
        elif step.kind == "tool_call":
            args = json.dumps(step.tool_input or {}, indent=2, default=str)
            self.console.print(
                Panel(
                    Syntax(args, "json", theme="ansi_dark", line_numbers=False),
                    title=f"[bold yellow]→ {step.tool_name}[/bold yellow]",
                    border_style="yellow",
                )
            )
        elif step.kind == "tool_result":
            result = step.tool_result or {}
            ok = result.get("ok", True)
            colour = "green" if ok else "red"
            preview = _preview(result)
            self.console.print(
                Panel(
                    Syntax(preview, "json", theme="ansi_dark", line_numbers=False),
                    title=f"[bold {colour}]← result[/bold {colour}]",
                    border_style=colour,
                )
            )
        elif step.kind == "stop":
            final = (step.content or "").strip()
            if final:
                self.console.print(
                    Panel.fit(
                        final,
                        title="[bold green]final[/bold green]",
                        border_style="green",
                    )
                )


def console_confirm(console: Console) -> PermissionCallback:
    """Return a ``PermissionCallback`` that asks the user via rich prompt."""

    def _ask(tool_name: str, arguments: dict[str, Any]) -> bool:
        args = json.dumps(arguments, indent=2, default=str)
        console.print(
            Panel(
                Syntax(args, "json", theme="ansi_dark", line_numbers=False),
                title=f"[bold magenta]Permission requested: {tool_name}[/bold magenta]",
                border_style="magenta",
            )
        )
        return Confirm.ask(
            f"Allow [bold]{tool_name}[/bold]?",
            default=False,
            console=console,
        )

    return _ask


def auto_approve(_tool_name: str, _arguments: dict[str, Any]) -> bool:
    """Permission callback that always allows — for ``--auto-approve`` runs."""
    return True


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _preview(payload: Any, *, limit: int = 2000) -> str:
    """Pretty-print a payload, truncating to keep the panel skim-able."""
    text = json.dumps(payload, indent=2, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated, {len(text) - limit} more chars]"


# Type used by the orchestrator to receive a permission callable.
PermissionCallableFactory = Callable[[Console], PermissionCallback]
