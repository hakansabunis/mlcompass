"""Pluggable agent backends.

Each backend implements :class:`AgentBackend` and runs the LLM loop
behind a uniform interface. The orchestrator picks one by name.
"""

from __future__ import annotations

from ._common import (
    AgentBackend,
    AgentResult,
    AgentStep,
    PermissionCallback,
    StepCallback,
)

__all__ = [
    "AgentBackend",
    "AgentResult",
    "AgentStep",
    "PermissionCallback",
    "StepCallback",
]
