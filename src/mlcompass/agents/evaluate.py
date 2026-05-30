"""Optional LLM layer for the ``evaluate`` command.

Given a structured evaluation result, asks Claude to interpret what the
numbers mean: where the model is strong, where it breaks down, what to
investigate next, and (for classification) which threshold to ship.
"""

from __future__ import annotations

import json
from typing import Any

from agentlite import Agent

from ._common import AgentResponseError, parse_json_response

EVALUATE_MODEL_DEFAULT = "claude-opus-4-7"

EVALUATE_INTERPRETER_PROMPT = """You are a senior ML engineer reviewing a colleague's post-training evaluation report.

You will receive a JSON object that contains the deterministic metrics, confusion matrix, threshold sweep (for binary classification), per-class breakdown (for multiclass), residual stats (for regression), hard examples, and any warnings the analyzer raised.

Your job:

1. Produce an **assessment** — one paragraph summarising what the numbers say about the model overall.
2. List 2-5 **strengths**: things the model is doing well. Cite specific numbers.
3. List 2-5 **weaknesses**: where the model is breaking down. Cite specific rows or classes when you can.
4. Propose 1-3 **next_steps** the user could take right now — concrete enough to act on. Examples: "ship threshold 0.55", "look at hard example 42, label is probably mis-annotated", "add features for class C, currently F1=0.18".

Reply with a single JSON object — no preamble, no markdown fences — matching this shape:

{
  "assessment": "Strong recall at moderate precision; threshold is too aggressive for production.",
  "strengths": [
    "AUC 0.94 — model separates classes well at the ranking level.",
    "Recall 0.95 — almost no positives are missed."
  ],
  "weaknesses": [
    "Precision 0.62 at threshold 0.5 — most flagged items are not actually positives.",
    "Hard example 42 (prob 0.97, label 0) suggests a borderline case the model rotates around."
  ],
  "next_steps": [
    "Ship threshold 0.65 — peaks F1 at 0.83 in the sweep with little recall sacrifice.",
    "Inspect the 5 hard examples; if any are mislabelled, fix and retrain."
  ]
}

Rules:
- Cite numbers and rows that appear in the input. Don't invent metrics or class names.
- For binary classification, lean on the threshold sweep — usually the most actionable insight.
- For regression, point at the residual block and hard examples.
- If warnings is empty and the metrics look fine, say so plainly in the assessment."""


class EvaluateAgentError(AgentResponseError):
    """Raised when the evaluate interpreter returns malformed JSON."""


# --------------------------------------------------------------------------- #
# Agent construction                                                          #
# --------------------------------------------------------------------------- #


def build_evaluate_agent(
    *,
    client: Any | None = None,
    model: str = EVALUATE_MODEL_DEFAULT,
) -> Agent:
    """Build the evaluation interpreter agent (pure reasoner, no tools)."""
    return Agent(
        model=model,
        system=EVALUATE_INTERPRETER_PROMPT,
        tools=[],
        client=client,
        max_turns=2,
    )


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def interpret_evaluation(
    evaluation: dict[str, Any],
    *,
    client: Any | None = None,
    model: str = EVALUATE_MODEL_DEFAULT,
) -> dict[str, Any]:
    """Run the interpreter on a structured evaluation result.

    Returns:
        Parsed ``{assessment, strengths, weaknesses, next_steps}`` dict.

    Raises:
        EvaluateAgentError: If the response shape is wrong.
    """
    agent = build_evaluate_agent(client=client, model=model)

    user_message = (
        "Here is the evaluation report from mlcompass evaluate. Please "
        "interpret it for me.\n\n"
        f"```json\n{json.dumps(evaluation, indent=2, default=str)}\n```"
    )

    raw = agent.run(user_message)
    return parse_json_response(
        raw,
        required_keys=("assessment", "strengths", "weaknesses", "next_steps"),
        error_class=EvaluateAgentError,
    )
