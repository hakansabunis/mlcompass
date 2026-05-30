"""Agent-run journal — per-run transcript on disk.

Every ``mlcompass agent`` invocation writes its events to
``.mlcompass/agent_runs/<run-id>/transcript.jsonl`` plus a
``summary.md`` with the final answer. The run-id is a short
hash + timestamp so two runs on the same day don't collide.

This is a tiny audit trail: the orchestrator streams ``AgentStep``
events to the UI in real time, and the writer mirrors each event to
disk so we can reconstruct what the agent did long after the
terminal session closed.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .backends import AgentResult, AgentStep


class TranscriptWriter:
    """Append-only JSONL writer for one agent run."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: TextIO | None = None

    def __enter__(self) -> TranscriptWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc: object) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def write_step(self, step: AgentStep) -> None:
        """Append one ``AgentStep`` as a JSON line."""
        if self._fh is None:
            raise RuntimeError("TranscriptWriter must be used as a context manager")
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **asdict(step),
        }
        # Truncate giant tool_result payloads in the on-disk transcript
        # so the journal stays grep-friendly. Full payloads are still
        # visible in the live UI.
        if record.get("tool_result") is not None:
            record["tool_result"] = _truncate(record["tool_result"], limit=4096)
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()


def new_run_dir(project_path: Path) -> Path:
    """Return a fresh ``.mlcompass/agent_runs/<id>/`` directory."""
    runs = project_path / "agent_runs"
    runs.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    base = now.strftime("%Y%m%d-%H%M%S")
    run_dir = runs / base
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = runs / f"{base}-{suffix:02d}"
    run_dir.mkdir()
    return run_dir


def write_summary(
    run_dir: Path,
    *,
    task: str,
    result: AgentResult,
    backend_name: str,
    model: str,
) -> Path:
    """Write a human-readable ``summary.md`` next to ``transcript.jsonl``.

    The summary captures: the original task, the backend + model used,
    the final agent text, and stop-reason / turn count metadata so
    someone reviewing the run later can see how the agent terminated
    without having to read every event.
    """
    summary_path = run_dir / "summary.md"
    lines = [
        f"# mlcompass agent — {run_dir.name}",
        "",
        f"- **Backend**: `{backend_name}`",
        f"- **Model**: `{model}`",
        f"- **Stop reason**: `{result.stop_reason}`",
        f"- **Turns**: {result.turns}",
        f"- **Ok**: {result.ok}",
    ]
    if result.error:
        lines.append(f"- **Error**: {result.error}")
    lines.extend(
        [
            "",
            "## Task",
            "",
            task.strip(),
            "",
            "## Final answer",
            "",
            result.final_text.strip() or "_(no final text)_",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _truncate(payload: Any, *, limit: int) -> Any:
    """Cap a JSON-safe payload so a single tool result doesn't blow up the log."""
    s = json.dumps(payload, default=str)
    if len(s) <= limit:
        return payload
    return {
        "_truncated": True,
        "_original_size": len(s),
        "_preview": s[:limit],
    }
