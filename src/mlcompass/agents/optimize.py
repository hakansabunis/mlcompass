"""Optional ``--llm`` strategist for the ``optimize`` command.

Given a deterministic optimize report (leader, sensitivity,
suggestions), asks Claude for a richer plan:

- a headline summary,
- the pattern the strategist sees in the run history (e.g. "lower
  learning rate steadily helps, but batch size has hit a ceiling"),
- a 3-step concrete plan for the user's next sprint.

Strategist output complements (does not replace) the deterministic
suggestions in :func:`mlcompass.tools.optimize.optimize_history`.
"""

from __future__ import annotations

import json
from typing import Any

from agentlite import Agent

from ._common import AgentResponseError, parse_json_response

OPTIMIZE_MODEL_DEFAULT = "claude-opus-4-7"

OPTIMIZE_STRATEGIST_PROMPT = """You are mlcompass-optimize, a hyperparameter-tuning strategist.

You will receive a JSON object describing the run history of a training
project: the best run, a leaderboard, per-hyperparameter sensitivity
(rank correlation with the chosen metric), and a few deterministic
"next config" suggestions.

Your job:

1. Read the leaderboard and the sensitivity table.
2. Spot the dominant pattern (which knobs matter, monotone vs U-shape,
   ceiling, noise floor).
3. Reply with a single JSON object — no preamble, no markdown fences —
   matching this shape:

{
  "headline": "one-sentence summary, max 25 words",
  "pattern": "1-2 sentence description of what the history is telling you",
  "next_plan": [
    "concrete step 1 (≤ 25 words)",
    "step 2",
    "step 3"
  ]
}

Rules:
- Mention hyperparameters by name.
- "next_plan" must have 2-4 entries.
- If there is too little signal (e.g. only 1-2 runs), say so plainly
  in headline + pattern and recommend more runs in next_plan.
- Output ONLY the JSON object."""


class OptimizeAgentError(AgentResponseError):
    """Raised when the optimize strategist returns malformed JSON."""


def build_optimize_agent(
    *,
    client: Any | None = None,
    model: str = OPTIMIZE_MODEL_DEFAULT,
) -> Agent:
    """Build the optimize strategist agent (pure reasoner, no tools)."""
    return Agent(
        model=model,
        system=OPTIMIZE_STRATEGIST_PROMPT,
        tools=[],
        client=client,
        max_turns=2,
    )


def strategize_optimize(
    report: dict[str, Any],
    *,
    client: Any | None = None,
    model: str = OPTIMIZE_MODEL_DEFAULT,
) -> dict[str, Any]:
    """Ask Claude to plan the next HPO sprint."""
    agent = build_optimize_agent(client=client, model=model)
    payload = json.dumps(_compact(report), default=str)
    user_message = (
        "Here is the optimize report from mlcompass optimize. Please give "
        "a headline, the dominant pattern, and a 2-4 step plan.\n\n"
        f"```json\n{payload}\n```"
    )
    raw = agent.run(user_message)
    parsed = parse_json_response(
        raw,
        required_keys=("headline", "pattern", "next_plan"),
        error_class=OptimizeAgentError,
    )
    return {
        "headline": str(parsed.get("headline", "")).strip(),
        "pattern": str(parsed.get("pattern", "")).strip(),
        "next_plan": [str(s).strip() for s in (parsed.get("next_plan") or []) if s],
    }


def _compact(report: dict[str, Any]) -> dict[str, Any]:
    """Drop verbose details so the prompt fits comfortably."""
    return {
        "metric": report["metric"],
        "direction": report["direction"],
        "n_runs": report["n_runs"],
        "n_scored": report["n_scored"],
        "best": report["best"],
        "leaderboard": report["leaderboard"],
        "sensitivity": report["sensitivity"],
        "suggestions": report["suggestions"],
    }
