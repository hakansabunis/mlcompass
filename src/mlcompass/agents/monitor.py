"""Optional ``--llm`` interpreter for drift reports.

Takes a structured drift report (the output of
:func:`mlcompass.tools.drift.detect_drift`) and asks Claude to:

- write a one-sentence headline,
- name the most likely cause of the drift, and
- propose 2-4 concrete next steps the user can take.

The deterministic drift detector remains the source of truth; this
layer just gives the user a readable interpretation when they pass
``--llm``.
"""

from __future__ import annotations

import json
from typing import Any

from agentlite import Agent

from ._common import AgentResponseError, parse_json_response

MONITOR_MODEL_DEFAULT = "claude-opus-4-7"

MONITOR_INTERPRETER_PROMPT = """You are mlcompass-monitor, a drift interpretation assistant.

You will receive a JSON object describing the result of a drift check that
compared a reference dataset (what a model was trained on) to a current
dataset (what it's seeing now).

Your job:

1. Read the aggregate signals (mean / max PSI, severity counts).
2. Read the per-feature breakdown and the top-drifted list.
3. Reply with a single JSON object — no preamble, no markdown fences —
   matching this shape:

{
  "headline": "one-sentence summary, max 25 words",
  "likely_cause": "1-2 sentence guess at WHY the drift happened (e.g., upstream pipeline change, seasonality, sampling bias)",
  "next_steps": [
    "concrete action (≤ 20 words)",
    "second action",
    "..."
  ]
}

Rules:
- Be specific. Mention the most-drifted feature names by name when relevant.
- If the verdict says "stable", say so plainly — do NOT invent drift.
- "next_steps" must have 2-4 entries.
- Output ONLY the JSON object."""


class MonitorAgentError(AgentResponseError):
    """Raised when the drift interpreter returns malformed JSON."""


def build_monitor_agent(
    *,
    client: Any | None = None,
    model: str = MONITOR_MODEL_DEFAULT,
) -> Agent:
    """Build the drift interpreter agent (pure reasoner, no tools)."""
    return Agent(
        model=model,
        system=MONITOR_INTERPRETER_PROMPT,
        tools=[],
        client=client,
        max_turns=2,
    )


def interpret_drift(
    report: dict[str, Any],
    *,
    client: Any | None = None,
    model: str = MONITOR_MODEL_DEFAULT,
) -> dict[str, Any]:
    """Ask Claude to interpret a deterministic drift report."""
    agent = build_monitor_agent(client=client, model=model)
    payload = json.dumps(_compact(report), default=str)

    user_message = (
        "Here is the drift report from mlcompass monitor. Please write a "
        "headline, the most likely cause, and 2-4 concrete next steps.\n\n"
        f"```json\n{payload}\n```"
    )
    raw = agent.run(user_message)
    parsed = parse_json_response(
        raw,
        required_keys=("headline", "likely_cause", "next_steps"),
        error_class=MonitorAgentError,
    )

    return {
        "headline": str(parsed.get("headline", "")).strip(),
        "likely_cause": str(parsed.get("likely_cause", "")).strip(),
        "next_steps": [str(s).strip() for s in (parsed.get("next_steps") or []) if s],
    }


def _compact(report: dict[str, Any]) -> dict[str, Any]:
    """Drop verbose per-feature warning text so the prompt stays small."""
    keep_feature = ("feature", "kind", "psi", "severity", "ks_stat", "chi2_stat")
    return {
        "reference_rows": report["reference_rows"],
        "current_rows": report["current_rows"],
        "features": report["features"],
        "aggregate": report["aggregate"],
        "top_drifted": report["top_drifted"],
        "feature_results": [
            {k: row.get(k) for k in keep_feature} for row in report["feature_results"]
        ],
        "verdict": report["verdict"],
    }
