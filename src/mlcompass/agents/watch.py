"""Optional LLM layer for the ``watch`` command.

Given a deterministic anomaly report (snapshots + findings), asks
Claude to hypothesize the *cause* of each finding and recommend a
concrete next action.

This is the layer most people will want — anomaly detectors say
"loss plateaued", the diagnostician explains *why* and *what to do*.
"""

from __future__ import annotations

import json
from typing import Any

from agentlite import Agent

from ._common import AgentResponseError, parse_json_response

WATCH_MODEL_DEFAULT = "claude-opus-4-7"

WATCH_DIAGNOSTICIAN_PROMPT = """You are a senior ML engineer pair-debugging a training run with a colleague who just sent you the metric history.

You will receive a JSON object with two keys:
- `snapshots`: ordered metric history (epoch, step, per-metric values)
- `findings`: the deterministic anomalies the analyzer surfaced

Your job:

1. For each finding, produce a **hypothesis** — the most likely root cause given the snapshot evidence. Be specific (cite epoch numbers and metric values when you can).
2. For each finding, give one **recommended_action** the user can take *right now*. Concrete enough to act on without further thought.
3. Tag each diagnosis with a confidence: "high", "medium", or "low".
4. End with a one-paragraph summary of the run's overall health.

Reply with a single JSON object — no preamble, no fences — matching this shape:

{
  "diagnosis": [
    {
      "finding_rule_id": "overfitting",
      "hypothesis": "Train loss kept falling through epoch 7 while val loss reversed at epoch 4; capacity is too high for the dataset size.",
      "recommended_action": "Bump dropout from 0.1 to 0.3 and restart from the epoch-4 checkpoint.",
      "confidence": "high"
    }
  ],
  "summary": "Overfit started at epoch 4. Stop training, regularize, restart."
}

Rules:
- Include exactly one diagnosis entry per finding in the input.
- Match `finding_rule_id` to the rule_id field in the input findings.
- Don't invent metrics or epochs that aren't in the snapshots."""


class WatchAgentError(AgentResponseError):
    """Raised when the watch diagnostician returns malformed JSON."""


def build_watch_agent(
    *,
    client: Any | None = None,
    model: str = WATCH_MODEL_DEFAULT,
) -> Agent:
    """Build the watch diagnostician agent (pure reasoner, no tools)."""
    return Agent(
        model=model,
        system=WATCH_DIAGNOSTICIAN_PROMPT,
        tools=[],
        client=client,
        max_turns=2,
    )


def diagnose_findings(
    snapshots: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    *,
    client: Any | None = None,
    model: str = WATCH_MODEL_DEFAULT,
) -> dict[str, Any]:
    """Run the diagnostician on a deterministic watch result.

    ``snapshots`` should be a list of plain dicts (the test surface
    accepts dataclass-as-dict conversions equally well; the agent only
    sees JSON either way).

    Returns the parsed ``{diagnosis, summary}`` dict.

    Raises:
        WatchAgentError: If the response shape is wrong.
    """
    if not findings:
        return {"diagnosis": [], "summary": "No anomalies; training looks healthy."}

    agent = build_watch_agent(client=client, model=model)

    payload = {"snapshots": snapshots, "findings": findings}
    user_message = (
        "Here is the watch report from mlcompass. Please diagnose each "
        "finding and recommend a concrete action.\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )

    raw = agent.run(user_message)
    return parse_json_response(
        raw,
        required_keys=("diagnosis", "summary"),
        error_class=WatchAgentError,
    )
