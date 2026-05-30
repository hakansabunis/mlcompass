"""Optional LLM layer for the ``compare`` command.

Given a structured run-vs-run comparison (config diff, metric diff,
verdict), asks Claude to explain *why* the better run was better and
propose the next experiment.
"""

from __future__ import annotations

import json
from typing import Any

from agentlite import Agent

from ._common import AgentResponseError, parse_json_response

COMPARE_MODEL_DEFAULT = "claude-opus-4-7"

COMPARE_HYPOTHESIZER_PROMPT = """You are a senior ML engineer doing a post-mortem on two training runs with a colleague.

You will receive a JSON object containing:
- summary metadata for each run (id, name, final metrics, config)
- a deterministic config diff
- a deterministic per-metric comparison
- the overall verdict the analyzer assigned

Your job:

1. Explain the verdict: **why** the winning run won (or why the result is mixed). Cite specific config and metric values from the input.
2. Identify the **key_factors** — the small set of config changes that most likely drove the outcome. For each, give your confidence ("high", "medium", "low") and a one-sentence reason.
3. Propose **next_experiment** — the next single config tweak the user should try, based on what these two runs revealed.

Reply with a single JSON object — no preamble, no markdown fences — matching this shape:

{
  "hypothesis": "Run B's lower learning rate prevented the late-epoch instability visible in Run A's val_loss bump at epoch 6.",
  "key_factors": [
    {
      "config_key": "lr",
      "impact": "high",
      "reason": "Lower LR usually stabilises late training, matching the val_loss curve here."
    }
  ],
  "next_experiment": "Re-run B with one more dropout step (0.3 -> 0.4) to see if it still improves at the same lr."
}

Rules:
- Cite values that appear in the input. Don't invent metrics.
- If the verdict is "mixed" or "inconclusive", say so plainly in the hypothesis.
- "next_experiment" is a single sentence describing one change."""


class CompareAgentError(AgentResponseError):
    """Raised when the compare hypothesizer returns malformed JSON."""


def build_compare_agent(
    *,
    client: Any | None = None,
    model: str = COMPARE_MODEL_DEFAULT,
) -> Agent:
    """Build the compare hypothesizer agent (pure reasoner, no tools)."""
    return Agent(
        model=model,
        system=COMPARE_HYPOTHESIZER_PROMPT,
        tools=[],
        client=client,
        max_turns=2,
    )


def hypothesize_comparison(
    comparison: dict[str, Any],
    *,
    client: Any | None = None,
    model: str = COMPARE_MODEL_DEFAULT,
) -> dict[str, Any]:
    """Run the hypothesizer on a deterministic comparison.

    Returns the parsed ``{hypothesis, key_factors, next_experiment}`` dict.

    Raises:
        CompareAgentError: If the response shape is wrong.
    """
    agent = build_compare_agent(client=client, model=model)

    user_message = (
        "Here is the side-by-side comparison from mlcompass compare. "
        "Please write the hypothesis, identify key factors, and propose "
        "the next experiment.\n\n"
        f"```json\n{json.dumps(comparison, indent=2)}\n```"
    )

    raw = agent.run(user_message)
    return parse_json_response(
        raw,
        required_keys=("hypothesis", "key_factors", "next_experiment"),
        error_class=CompareAgentError,
    )
