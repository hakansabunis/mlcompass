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

from ._common import (
    AgentResponseError,
    build_anthropic_agent,
    parse_json_response,
    resolve_llm_config,
    run_json_agent,
)

ADVISOR_MODEL_DEFAULT = "claude-opus-4-7"

ADVISOR_SYSTEM_PROMPT = """You are a senior data scientist advising a colleague who just analyzed a new dataset. Your job is to recommend the next steps.

You will receive a structured JSON dataset analysis. Based on it, produce:

1. **Top 3 model families** to try, with one-line reasoning.

   Do NOT predict a metric value or range. Nothing has been trained, so
   any number you gave would be invented, and this tool does not print
   invented numbers next to measured ones.
2. **Feature engineering suggestions**: per-column or cross-column hints
   that have a high likelihood of helping.
3. **Pitfalls**: data-quality or methodology issues the user should mitigate
   before training.

Format your reply as a single JSON object matching this schema, and only
that — no preamble, no markdown fence:

{
  "models": [
    {"name": "XGBoost", "reason": "..."}
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
- Never state or imply an expected metric value. No model has been trained.
- Cite the column or fact you reasoned from inside the ``reason`` field.
- When uncertain, prefer conservative, well-established choices."""


# --------------------------------------------------------------------------- #
# Agent construction                                                          #
# --------------------------------------------------------------------------- #


def build_advisor_agent(
    *,
    client: Any | None = None,
    model: str | None = None,
) -> Any:
    """Build the Anthropic model + feature engineering advisor agent.

    Args:
        client: Optional Anthropic client or ``MockClient`` for tests.
            When ``None``, agentlite creates a real client from
            ``ANTHROPIC_API_KEY``.
        model: Claude model name, or None to resolve from
            ``MLCOMPASS_LLM_MODEL`` and then :data:`ADVISOR_MODEL_DEFAULT`.
            Opus is the default because the advisor's reasoning needs to be
            sharp; Haiku tends to oversimplify on tabular ML recommendations.

    Returns:
        A configured ``agentlite.Agent`` ready to accept a dataset analysis
        as its user message. This is the Anthropic path only; for an
        OpenAI-compatible endpoint call :func:`get_recommendation` with
        ``provider="openai"``.
    """
    resolved = resolve_llm_config(
        default_model=ADVISOR_MODEL_DEFAULT,
        provider="anthropic",
        model=model,
    )
    return build_anthropic_agent(
        system=ADVISOR_SYSTEM_PROMPT,
        model=resolved.model,
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
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run the advisor on a pre-computed dataset analysis.

    Provider, model, base URL and API key each resolve in the same order:
    **the explicit argument, then the environment, then the built-in
    default.** The environment variables are ``MLCOMPASS_LLM_PROVIDER``,
    ``MLCOMPASS_LLM_MODEL``, ``MLCOMPASS_LLM_BASE_URL``, and the provider's
    own key variable (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``). With
    none of them set the advisor talks to Anthropic exactly as it always
    has.

    Args:
        analysis: Output of ``tools.dataset.analyze_dataset()``.
        client: Optional pre-built provider client (tests inject fakes).
        model: Model identifier. Defaults to :data:`ADVISOR_MODEL_DEFAULT`
            on Anthropic; required for any other provider.
        provider: ``"anthropic"`` (default) or ``"openai"``. The OpenAI
            path is plain chat completions, so it covers any
            OpenAI-compatible endpoint — a local ollama or vLLM server
            included.
        base_url: Endpoint override, e.g. ``http://localhost:11434/v1``.
        api_key: Credential override. A local endpoint needs none.

    Returns:
        Parsed recommendation dict with keys ``models``, ``features``,
        ``pitfalls``.

    Raises:
        AdvisorParseError: If the agent's response can't be parsed as the
            expected JSON shape.
    """
    user_message = (
        "Here is the analysis of a dataset. Please produce model, feature, "
        "and pitfall recommendations as specified.\n\n"
        f"```json\n{json.dumps(analysis, indent=2)}\n```"
    )

    raw_response = run_json_agent(
        system=ADVISOR_SYSTEM_PROMPT,
        user=user_message,
        default_model=ADVISOR_MODEL_DEFAULT,
        client=client,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    return _parse_advisor_response(raw_response)


# --------------------------------------------------------------------------- #
# Internal: response parsing                                                  #
# --------------------------------------------------------------------------- #


class AdvisorParseError(AgentResponseError):
    """Raised when the advisor returns something other than valid JSON."""


def _parse_advisor_response(text: str) -> dict[str, Any]:
    """Strip optional markdown fences and parse JSON.

    The advisor predates :func:`_common.parse_json_response` and used to
    carry its own copy of that logic. It now delegates, so there is one
    fence-stripping / empty-reply / shape-check implementation for all nine
    agents. Light shape validation only — the schema can still evolve.
    """
    return parse_json_response(
        text,
        required_keys=("models", "features", "pitfalls"),
        error_class=AdvisorParseError,
    )
