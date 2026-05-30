"""Agent orchestrator — public entry point for the self-driving layer.

Wraps a backend with the on-disk transcript writer and an optional
streaming UI. The CLI subcommand (``mlcompass agent``) is a thin
adapter around :func:`run_agent`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from ..context import ProjectContext, ProjectNotFoundError
from .backends import AgentBackend, AgentResult, AgentStep, PermissionCallback
from .backends.anthropic_api import AnthropicAPIBackend
from .backends.claude_code import ClaudeCodeBackend
from .transcript import TranscriptWriter, new_run_dir, write_summary
from .ui import StepRenderer, auto_approve, console_confirm

_LOG = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("MLCOMPASS_AGENT_MODEL", "claude-sonnet-4-5")
DEFAULT_MAX_TURNS = 20


@dataclass
class AgentRunSummary:
    """Returned by :func:`run_agent`."""

    result: AgentResult
    run_dir: Path
    transcript_path: Path
    summary_path: Path


# --------------------------------------------------------------------------- #
# Public entry                                                                #
# --------------------------------------------------------------------------- #


def run_agent(
    task: str,
    *,
    project_path: str | os.PathLike[str] = ".",
    backend: str = "api",
    model: str = DEFAULT_MODEL,
    max_turns: int = DEFAULT_MAX_TURNS,
    auto_approve_mutations: bool = False,
    console: Console | None = None,
    extra_on_step: Callable[[AgentStep], None] | None = None,
) -> AgentRunSummary:
    """Drive the chosen backend until it converges or hits ``max_turns``.

    Args:
        task: Plain-language task description from the user.
        project_path: Where the mlcompass project lives (walks up
            looking for ``.mlcompass/``). If no project is found, the
            agent still runs but the transcript is written under
            ``<project_path>/.mlcompass-agent/`` so the audit trail is
            preserved.
        backend: ``"api"`` or ``"claude-code"``.
        model: Model name passed to the backend.
        max_turns: Hard cap on conversation turns.
        auto_approve_mutations: If ``True``, mutating tools (``init``)
            run without the confirm prompt. Use for CI / headless.
        console: Optional ``rich`` console. A fresh one is created if
            omitted so the function is safe to call from any context.
        extra_on_step: Optional additional ``StepCallback`` (e.g. test
            spy). Fires after the built-in UI + transcript writers.

    Returns:
        :class:`AgentRunSummary` with the backend's result plus paths
        to the on-disk transcript and human-readable summary.
    """
    console = console or Console()
    project_path = Path(project_path).resolve()

    project_root = _resolve_project_root(project_path)
    run_dir = new_run_dir(project_root)
    transcript_path = run_dir / "transcript.jsonl"

    backend_impl = _get_backend(backend)
    permission: PermissionCallback = (
        auto_approve if auto_approve_mutations else console_confirm(console)
    )
    renderer = StepRenderer(console)

    with TranscriptWriter(transcript_path) as writer:

        def on_step(step: AgentStep) -> None:
            renderer(step)
            try:
                writer.write_step(step)
            except OSError:  # pragma: no cover — defensive
                _LOG.exception("Failed to write transcript step")
            if extra_on_step is not None:
                extra_on_step(step)

        result = backend_impl.run(
            task=task,
            project_path=str(project_path),
            max_turns=max_turns,
            model=model,
            on_permission=permission,
            on_step=on_step,
        )

    summary_path = write_summary(
        run_dir,
        task=task,
        result=result,
        backend_name=backend_impl.name,
        model=model,
    )

    return AgentRunSummary(
        result=result,
        run_dir=run_dir,
        transcript_path=transcript_path,
        summary_path=summary_path,
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


_BACKENDS: dict[str, type[AgentBackend]] = {
    "api": AnthropicAPIBackend,
    "claude-code": ClaudeCodeBackend,
}


def _get_backend(name: str) -> AgentBackend:
    try:
        cls = _BACKENDS[name]
    except KeyError as e:
        valid = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"Unknown backend {name!r}. Available: {valid}") from e
    return cls()


def _resolve_project_root(project_path: Path) -> Path:
    """Return the ``.mlcompass/`` to use for the transcript.

    If a project already exists at or above ``project_path``, transcript
    files live inside it. Otherwise we fall back to a sibling
    ``.mlcompass-agent/`` directory so we don't silently create a
    half-initialised project (the user has to call
    ``mlcompass init`` for that).
    """
    try:
        project = ProjectContext.load(search_from=project_path)
    except ProjectNotFoundError:
        fallback = project_path / ".mlcompass-agent"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    return project.path


__all__ = [
    "AgentRunSummary",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_MODEL",
    "run_agent",
]
