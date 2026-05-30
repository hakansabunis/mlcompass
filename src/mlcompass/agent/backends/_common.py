"""Shared backend types — uniform contract every backend implements."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

# What kind of event a backend streams to the UI.
#
# - ``message``     model emitted a text block.
# - ``tool_call``   model asked to use a tool.
# - ``tool_result`` dispatcher returned a result (or a permission denial).
# - ``thinking``    extended-thinking trace block (anthropic >=0.50).
# - ``stop``        backend reached a terminal state (final answer / max
#                   turns / error). Carries the final text in ``content``.
StepKind = Literal["message", "tool_call", "tool_result", "thinking", "stop"]


@dataclass
class AgentStep:
    """One event in the streamed transcript."""

    kind: StepKind
    content: Any = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None


@dataclass
class AgentResult:
    """Final summary returned by the backend's ``run`` method."""

    ok: bool
    turns: int
    final_text: str
    stop_reason: str
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# Callbacks the orchestrator hands the backend.
#
# - ``PermissionCallback`` is called BEFORE a mutating tool dispatches.
#   Return ``True`` to allow, ``False`` to deny. The backend feeds the
#   denial back to the model as a ``tool_result`` so the model can
#   adapt or abort.
# - ``StepCallback`` is called for every event in real time. The
#   orchestrator's UI renders the event and the transcript writer
#   appends it to disk.
PermissionCallback = Callable[[str, dict[str, Any]], bool]
StepCallback = Callable[[AgentStep], None]


class AgentBackend(Protocol):
    """Interface every backend implements."""

    name: str

    def run(
        self,
        task: str,
        *,
        project_path: str,
        max_turns: int,
        model: str,
        on_permission: PermissionCallback,
        on_step: StepCallback,
        system_prompt: str | None = None,
    ) -> AgentResult: ...
