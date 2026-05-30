"""Optional LLM layer for the ``deploy`` command.

Given a structured deployment-readiness report, asks Claude for a
production-readiness verdict, the three biggest blockers, and the
2-3 highest-leverage next actions.
"""

from __future__ import annotations

import json
from typing import Any

from agentlite import Agent

from ._common import AgentResponseError, parse_json_response

DEPLOY_MODEL_DEFAULT = "claude-opus-4-7"

DEPLOY_ADVISOR_PROMPT = """You are a senior MLOps engineer reviewing a production-readiness report for a colleague's model.

You will receive a JSON object that contains the model metadata, the dependency manifest (or null), the target environment, a production checklist of {item, status, detail} rows, and any warnings raised by the static analyzer.

Your job:

1. Produce a one-paragraph **verdict** on production readiness, mentioning the target environment.
2. List up to three **blockers** — the things that absolutely have to be fixed before shipping. Empty list if nothing is blocking.
3. Propose up to three **next_steps** the team should take this week. Concrete, actionable, ordered by impact.
4. Suggest one **rollout_strategy** (e.g. shadow traffic, canary 1% → 10% → 50%, blue/green, batch-only first).

Reply with a single JSON object — no preamble, no fences — matching this shape:

{
  "verdict": "Almost ready for canary deploy on Lambda; size and format are fine but the dependency drift risk needs fixing first.",
  "blockers": [
    "Pin `requests` — unpinned production dependencies cause silent regressions."
  ],
  "next_steps": [
    "Lock all four pip dependencies to exact versions and rebuild the image.",
    "Measure cold-start latency on Lambda with the production wheel."
  ],
  "rollout_strategy": "Canary at 1% of traffic for 24h, then 10%, then 50%, then 100% if drift metrics stay clean."
}

Rules:
- Cite specific fields from the input (target name, format, manifest type, exact warnings). Don't invent facts.
- If there are zero warnings and the checklist is clean, say so plainly and propose monitoring rather than fixes."""


class DeployAgentError(AgentResponseError):
    """Raised when the deploy advisor returns malformed JSON."""


# --------------------------------------------------------------------------- #
# Agent construction                                                          #
# --------------------------------------------------------------------------- #


def build_deploy_agent(
    *,
    client: Any | None = None,
    model: str = DEPLOY_MODEL_DEFAULT,
) -> Agent:
    """Build the deployment-advisor agent (pure reasoner, no tools)."""
    return Agent(
        model=model,
        system=DEPLOY_ADVISOR_PROMPT,
        tools=[],
        client=client,
        max_turns=2,
    )


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def advise_deployment(
    report: dict[str, Any],
    *,
    client: Any | None = None,
    model: str = DEPLOY_MODEL_DEFAULT,
) -> dict[str, Any]:
    """Run the deploy advisor on a structured readiness report.

    Returns:
        Parsed ``{verdict, blockers, next_steps, rollout_strategy}`` dict.

    Raises:
        DeployAgentError: If the response shape is wrong.
    """
    agent = build_deploy_agent(client=client, model=model)

    user_message = (
        "Here is the deployment-readiness report from mlcompass deploy. "
        "Please assess production readiness.\n\n"
        f"```json\n{json.dumps(report, indent=2, default=str)}\n```"
    )

    raw = agent.run(user_message)
    return parse_json_response(
        raw,
        required_keys=("verdict", "blockers", "next_steps", "rollout_strategy"),
        error_class=DeployAgentError,
    )
