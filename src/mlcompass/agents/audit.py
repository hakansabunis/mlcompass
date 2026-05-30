"""Optional LLM layer for the ``audit`` command.

Given a structured audit result (the deterministic AST findings),
asks Claude to rank the findings by blast radius and produce a short
human-friendly synthesis.

The deterministic analyzer is authoritative for *what* is wrong;
this layer adds *which one matters most and why*.
"""

from __future__ import annotations

import json
from typing import Any

from agentlite import Agent

from ._common import AgentResponseError, parse_json_response

AUDIT_MODEL_DEFAULT = "claude-opus-4-7"

AUDIT_PRIORITIZER_PROMPT = """You are a senior ML engineer reviewing a static-analysis report on a colleague's training script.

You will receive a JSON object that contains the analyzer's structured findings.

Your job:

1. Rank the findings by **blast radius** — how likely each one is to silently ruin a training run or hide a real bug. Most severe first.
2. For each finding, write one short sentence explaining the realistic worst case if it's ignored.
3. End with a one-paragraph synthesis the user can read in five seconds.

Reply with a single JSON object — no preamble, no markdown fences — matching this shape:

{
  "priorities": [
    {
      "rule_id": "seed",
      "priority_rank": 1,
      "blast_radius": "Results cannot be reproduced; you cannot tell if a change actually helped."
    }
  ],
  "synthesis": "Fix the missing seed first, then …"
}

Rules:
- The "priorities" list must include every finding from the input, each appearing exactly once.
- "priority_rank" is 1 for the most critical, 2 for the next, etc.
- Keep "blast_radius" to a single sentence."""


class AuditAgentError(AgentResponseError):
    """Raised when the audit prioritizer returns malformed JSON."""


def build_audit_agent(
    *,
    client: Any | None = None,
    model: str = AUDIT_MODEL_DEFAULT,
) -> Agent:
    """Build the audit prioritizer agent (pure reasoner, no tools)."""
    return Agent(
        model=model,
        system=AUDIT_PRIORITIZER_PROMPT,
        tools=[],
        client=client,
        max_turns=2,
    )


def prioritize_findings(
    audit_result: dict[str, Any],
    *,
    client: Any | None = None,
    model: str = AUDIT_MODEL_DEFAULT,
) -> dict[str, Any]:
    """Run the prioritizer on a deterministic audit result.

    Returns the parsed ``{priorities, synthesis}`` dict.

    Raises:
        AuditAgentError: If the response shape is wrong.
    """
    if not audit_result.get("findings"):
        return {"priorities": [], "synthesis": "No findings to prioritize."}

    agent = build_audit_agent(client=client, model=model)

    user_message = (
        "Here is the static-analysis result from mlcompass audit. Please "
        "rank the findings by blast radius and write a synthesis.\n\n"
        f"```json\n{json.dumps(audit_result, indent=2)}\n```"
    )

    raw = agent.run(user_message)
    return parse_json_response(
        raw,
        required_keys=("priorities", "synthesis"),
        error_class=AuditAgentError,
    )
