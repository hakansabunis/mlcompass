"""Model + feature engineering advisor.

Consumes the structured output of ``tools.dataset.analyze_dataset`` and
produces a JSON recommendation: models to try, feature engineering
suggestions, and pitfalls to mitigate.

The analysis is pre-computed by deterministic Python (no LLM cost), and
fed to the agent as the user message. The advisor agent itself does not
need tools — it is a pure reasoner over the structured input. This keeps
``advise`` fast, cheap, and predictable.
"""

from __future__ import annotations

import json
from typing import Any

from agentlite import Agent

ADVISOR_MODEL_DEFAULT = "claude-opus-4-7"

ADVISOR_SYSTEM_PROMPT = """You are a senior data scientist advising a colleague who just analyzed a new dataset. Your job is to recommend the next steps.

You will receive a structured JSON dataset analysis. Based on it, produce:

1. **Top 3 model families** to try, with one-line reasoning and a realistic
   metric range (AUC for classification, RMSE/MAE for regression).
2. **Feature engineering suggestions**: per-column or cross-column hints
   that have a high likelihood of helping.
3. **Pitfalls**: data-quality or methodology issues the user should mitigate
   before training.

Format your reply as a single JSON object matching this schema, and only
that — no preamble, no markdown fence:

{
  "models": [
    {"name": "XGBoost", "reason": "...", "expected_metric": "AUC 0.82 - 0.87"}
  ],
  "features": [
    {"column": "signup_date", "suggestion": "derive days_since_signup, month, dayofweek", "reason": "..."}
  ],
  "pitfalls": [
    {"issue": "Class imbalance (12% positive)", "mitigation": "Use AUC/F1, class_weight='balanced', or focal loss"}
  ]
}

Rules:
- Always include at least one interpretable baseline (logistic / linear
  regression, decision tree, etc.) so the user has a sanity check.
- Use realistic metric ranges based on the dataset signal — do not promise
  numbers you can't back up.
- Cite the column or fact you reasoned from inside the ``reason`` field.
- When uncertain, prefer conservative, well-established choices."""


# --------------------------------------------------------------------------- #
# Agent construction                                                          #
# --------------------------------------------------------------------------- #


def build_advisor_agent(
    *,
    client: Any | None = None,
    model: str = ADVISOR_MODEL_DEFAULT,
) -> Agent:
    """Build the model + feature engineering advisor agent.

    Args:
        client: Optional Anthropic client or ``MockClient`` for tests.
            When ``None``, agentlite creates a real client from
            ``ANTHROPIC_API_KEY``.
        model: Claude model name. Opus is the default because the advisor's
            reasoning needs to be sharp; Haiku tends to oversimplify on
            tabular ML recommendations.

    Returns:
        A configured ``agentlite.Agent`` ready to accept a dataset analysis
        as its user message.
    """
    return Agent(
        model=model,
        system=ADVISOR_SYSTEM_PROMPT,
        tools=[],  # Pure reasoner — no tool calls needed.
        client=client,
        max_turns=2,  # Sanity cap; single-turn is the expected case.
    )


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def get_recommendation(
    analysis: dict[str, Any],
    *,
    client: Any | None = None,
    model: str = ADVISOR_MODEL_DEFAULT,
) -> dict[str, Any]:
    """Run the advisor on a pre-computed dataset analysis.

    Args:
        analysis: Output of ``tools.dataset.analyze_dataset()``.
        client: Optional client for testing.
        model: Claude model name.

    Returns:
        Parsed recommendation dict with keys ``models``, ``features``,
        ``pitfalls``.

    Raises:
        AdvisorParseError: If the agent's response can't be parsed as the
            expected JSON shape.
    """
    agent = build_advisor_agent(client=client, model=model)

    user_message = (
        "Here is the analysis of a dataset. Please produce model, feature, "
        "and pitfall recommendations as specified.\n\n"
        f"```json\n{json.dumps(analysis, indent=2)}\n```"
    )

    raw_response = agent.run(user_message)
    return _parse_advisor_response(raw_response)


# --------------------------------------------------------------------------- #
# Internal: response parsing                                                  #
# --------------------------------------------------------------------------- #


class AdvisorParseError(ValueError):
    """Raised when the advisor returns something other than valid JSON."""


def _parse_advisor_response(text: str) -> dict[str, Any]:
    """Strip optional markdown fences and parse JSON."""
    stripped = text.strip()

    # The system prompt explicitly forbids fences, but be lenient.
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Drop opening fence (``` or ```json) and find the closing fence
        start = 1
        end = len(lines)
        for i, line in enumerate(lines[1:], 1):
            if line.strip().startswith("```"):
                end = i
                break
        stripped = "\n".join(lines[start:end])

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        snippet = text[:200].replace("\n", " ")
        raise AdvisorParseError(
            f"Advisor response was not valid JSON: {snippet!r}"
        ) from exc

    if not isinstance(parsed, dict):
        raise AdvisorParseError(
            f"Advisor response was JSON but not an object (got {type(parsed).__name__})"
        )

    # Light shape validation — keep it loose so the schema can evolve.
    for required_key in ("models", "features", "pitfalls"):
        if required_key not in parsed:
            raise AdvisorParseError(
                f"Advisor response missing required key '{required_key}'. "
                f"Got: {sorted(parsed.keys())}"
            )

    return parsed
