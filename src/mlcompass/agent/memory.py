"""Agent memory layer — Faz 8c.

Two complementary memory sources stitch into a single context block
the orchestrator prepends to every new agent run:

1. **Project decisions** — the rolling ``decisions[]`` list from
   ``.mlcompass/context.json``. Lightweight, always available, cheap
   to inject (one line per entry).
2. **Prior agent runs** — full transcripts under
   ``.mlcompass/agent_runs/<id>/transcript.jsonl``. These can be
   large; we summarise them with an ``agentlite`` sub-agent so the
   memory block stays under a few hundred tokens.

Both sources are optional. Missing transcripts / missing context.json
fall through to "no memory" without crashing.

The orchestrator's ``--resume <run-id>`` flag uses
:func:`load_run_summary` to seed the context with a single prior run.
Without ``--resume`` we offer :func:`load_recent_context` which gives
the agent a small running biography of the project.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

AGENT_RUNS_SUBDIR = "agent_runs"

# Hard cap on transcript bytes we hand to the summariser so a runaway
# run can't blow up the prompt budget.
_MAX_TRANSCRIPT_CHARS = 20_000


@dataclass
class MemoryBlock:
    """One assembled memory block ready to inject into a new run."""

    headline: str
    decisions_summary: str
    prior_run_summary: str
    source_run_id: str | None = None

    def as_prompt_prefix(self) -> str:
        """Render the memory block as the prefix for a new task prompt."""
        chunks: list[str] = []
        if self.headline:
            chunks.append(f"<memory_headline>{self.headline}</memory_headline>")
        if self.decisions_summary:
            chunks.append(f"<recent_decisions>\n{self.decisions_summary}\n</recent_decisions>")
        if self.prior_run_summary:
            tag = (
                f"prior_agent_run id={self.source_run_id!r}"
                if self.source_run_id
                else "prior_agent_run"
            )
            chunks.append(f"<{tag}>\n{self.prior_run_summary}\n</prior_agent_run>")
        return "\n\n".join(chunks)


# --------------------------------------------------------------------------- #
# Decisions reader                                                            #
# --------------------------------------------------------------------------- #


def load_decisions_summary(
    project_path: Path,
    *,
    limit: int = 10,
) -> str:
    """Return a compact text summary of the latest ``decisions[]`` entries."""
    context_path = project_path / "context.json"
    if not context_path.is_file():
        return ""
    try:
        data = json.loads(context_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    decisions = data.get("decisions") or []
    if not decisions:
        return ""
    tail = decisions[-limit:]
    lines = [
        f"- [{(d.get('timestamp') or '')[:10]}] {d.get('command')}: "
        f"{(d.get('summary') or '').strip()}"
        for d in tail
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Transcript reader + summariser                                              #
# --------------------------------------------------------------------------- #


def load_transcript(transcript_path: Path) -> list[dict[str, Any]]:
    """Parse a per-run JSONL transcript into a list of step dicts."""
    if not transcript_path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with transcript_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                _LOG.debug("Skipping unparseable transcript line: %s", line[:80])
    return out


def find_run_dir(project_path: Path, run_id: str) -> Path | None:
    """Return the on-disk path for a prior agent run, or ``None``."""
    base = project_path / AGENT_RUNS_SUBDIR
    if not base.is_dir():
        return None
    candidate = base / run_id
    if candidate.is_dir():
        return candidate
    # Also accept run-id prefixes (timestamp-only) for convenience.
    matches = sorted(p for p in base.iterdir() if p.name.startswith(run_id))
    return matches[0] if matches else None


def list_recent_runs(project_path: Path, *, limit: int = 5) -> list[Path]:
    """Return the ``limit`` most recent agent run directories."""
    base = project_path / AGENT_RUNS_SUBDIR
    if not base.is_dir():
        return []
    runs = sorted(
        (p for p in base.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    return runs[:limit]


def transcript_to_text(
    steps: Iterable[dict[str, Any]],
    *,
    max_chars: int = _MAX_TRANSCRIPT_CHARS,
) -> str:
    """Render a transcript as a compact human-readable text block.

    We elide oversized ``tool_result`` payloads to keep the prompt
    budget tight; the summariser only needs the gist of each step.
    """
    lines: list[str] = []
    used = 0
    for step in steps:
        kind = step.get("kind", "?")
        if kind == "message":
            line = f"agent: {_short(step.get('content'), 200)}"
        elif kind == "tool_call":
            args_short = json.dumps(step.get("tool_input") or {}, default=str)[:120]
            line = f"→ {step.get('tool_name')}({args_short})"
        elif kind == "tool_result":
            tr = step.get("tool_result") or {}
            ok = tr.get("ok", True)
            preview = json.dumps(tr, default=str)[:120]
            line = f"  ← {'ok' if ok else 'ERR'}: {preview}"
        elif kind == "stop":
            line = f"stop: {_short(step.get('content'), 200)}"
        else:
            continue
        used += len(line) + 1
        if used > max_chars:
            lines.append("[... transcript truncated ...]")
            break
        lines.append(line)
    return "\n".join(lines)


def summarise_transcript_with_agentlite(
    transcript_text: str,
    *,
    model: str = "claude-opus-4-7",
    client: Any | None = None,
) -> str:
    """Ask an agentlite-driven sub-agent to compress a transcript.

    Returns:
        A 100-200 word summary covering: original task, tools the agent
        called, key findings, final answer, any pending follow-ups.
        Falls back to a graceful "(summarisation failed: …)" string on
        any LLM / parsing error so the orchestrator stays running.
    """
    if not transcript_text.strip():
        return "(empty transcript)"

    try:
        from agentlite import Agent
    except ImportError:  # pragma: no cover
        return "(agentlite not installed; cannot summarise transcript)"

    system_prompt = (
        "You summarise mlcompass agent run transcripts for cross-session "
        "memory. Read the transcript and produce a 100-200 word "
        "narrative covering: the original task, which mlcompass_* tools "
        "the agent called, the key findings from those calls, the final "
        "answer the agent gave the user, and any unresolved follow-ups. "
        "Be concrete (mention file paths, metrics, tool names). Output "
        "plain prose — no preamble, no markdown headings, no JSON."
    )
    try:
        agent = Agent(
            model=model,
            system=system_prompt,
            tools=[],
            client=client,
            max_turns=2,
        )
        return str(
            agent.run(
                "Summarise the following transcript:\n\n"
                f"<transcript>\n{transcript_text}\n</transcript>"
            )
        ).strip()
    except Exception as e:  # noqa: BLE001
        _LOG.warning("Transcript summariser failed: %s", e)
        return f"(transcript summarisation failed: {e})"


# --------------------------------------------------------------------------- #
# High-level entry points                                                     #
# --------------------------------------------------------------------------- #


def load_run_summary(
    project_path: Path,
    run_id: str,
    *,
    summarise: bool = True,
    model: str = "claude-opus-4-7",
    client: Any | None = None,
) -> MemoryBlock:
    """Build a memory block resuming a specific past run.

    Args:
        project_path: Path to the ``.mlcompass/`` project root.
        run_id: Subdirectory name under ``agent_runs/`` (full or prefix).
        summarise: When ``True`` (default), hand the transcript to
            agentlite for compression. Set to ``False`` for tests or
            when you want the raw transcript text inline.
        model: Model name for the summariser.
        client: Optional Anthropic client (mostly for tests).

    Returns:
        A :class:`MemoryBlock`. ``source_run_id`` echoes the run we
        resumed; ``prior_run_summary`` is the compressed transcript.
    """
    run_dir = find_run_dir(project_path, run_id)
    if run_dir is None:
        return MemoryBlock(
            headline=f"(could not find prior run {run_id!r})",
            decisions_summary="",
            prior_run_summary="",
        )
    steps = load_transcript(run_dir / "transcript.jsonl")
    if not steps:
        return MemoryBlock(
            headline=f"(prior run {run_dir.name} has no transcript)",
            decisions_summary="",
            prior_run_summary="",
            source_run_id=run_dir.name,
        )
    text = transcript_to_text(steps)
    summary = (
        summarise_transcript_with_agentlite(text, model=model, client=client) if summarise else text
    )
    decisions = load_decisions_summary(project_path)
    return MemoryBlock(
        headline=(f"Resuming prior agent run {run_dir.name}: {len(steps)} steps recorded."),
        decisions_summary=decisions,
        prior_run_summary=summary,
        source_run_id=run_dir.name,
    )


def load_recent_context(
    project_path: Path,
    *,
    include_last_run: bool = True,
    summarise: bool = True,
    model: str = "claude-opus-4-7",
    client: Any | None = None,
) -> MemoryBlock:
    """Build a memory block from decisions log plus the latest run.

    This is the default cross-session context the orchestrator injects
    when the user runs ``mlcompass agent`` without ``--resume``.
    """
    decisions = load_decisions_summary(project_path)
    prior_summary = ""
    source_id: str | None = None
    if include_last_run:
        runs = list_recent_runs(project_path, limit=1)
        if runs:
            source_id = runs[0].name
            steps = load_transcript(runs[0] / "transcript.jsonl")
            if steps:
                text = transcript_to_text(steps)
                prior_summary = (
                    summarise_transcript_with_agentlite(text, model=model, client=client)
                    if summarise
                    else text
                )

    headline = ""
    if decisions or prior_summary:
        headline = "Cross-session memory: pulling in recent project decisions" + (
            " + last agent run." if prior_summary else "."
        )
    return MemoryBlock(
        headline=headline,
        decisions_summary=decisions,
        prior_run_summary=prior_summary,
        source_run_id=source_id,
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _short(value: Any, limit: int) -> str:
    if value is None:
        return ""
    s = str(value)
    if len(s) <= limit:
        return s
    return s[:limit] + "…"


__all__ = [
    "AGENT_RUNS_SUBDIR",
    "MemoryBlock",
    "find_run_dir",
    "list_recent_runs",
    "load_decisions_summary",
    "load_recent_context",
    "load_run_summary",
    "load_transcript",
    "summarise_transcript_with_agentlite",
    "transcript_to_text",
]
