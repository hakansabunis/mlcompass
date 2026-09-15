"""Optional LLM layer for the ``compare`` command.

Given a structured run-vs-run comparison (config diff, metric diff,
verdict), asks Claude to explain *why* the better run was better and
propose the next experiment.
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
    model: str | None = None,
) -> Any:
    """Build the Anthropic compare hypothesizer agent (pure reasoner, no tools)."""
    resolved = resolve_llm_config(
        default_model=COMPARE_MODEL_DEFAULT,
        provider="anthropic",
        model=model,
    )
    return build_anthropic_agent(
        system=COMPARE_HYPOTHESIZER_PROMPT,
        model=resolved.model,
        client=client,
        max_turns=2,
    )


def hypothesize_comparison(
    comparison: dict[str, Any],
    *,
    client: Any | None = None,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run the hypothesizer on a deterministic comparison.

    Provider, model, base URL and API key each resolve in the same order:
    **explicit argument, then environment (``MLCOMPASS_LLM_PROVIDER``,
    ``MLCOMPASS_LLM_MODEL``, ``MLCOMPASS_LLM_BASE_URL``, and the provider's
    own key variable), then the built-in default.** With none of them set
    the hypothesizer talks to Anthropic exactly as it always has.
    ``provider="openai"`` sends a chat completion instead, which reaches any
    OpenAI-compatible endpoint — including a local, keyless ollama server
    via ``base_url``.

    Returns the parsed ``{hypothesis, key_factors, next_experiment}`` dict.

    Raises:
        CompareAgentError: If the response shape is wrong.
    """
    user_message = (
        "Here is the side-by-side comparison from mlcompass compare. "
        "Please write the hypothesis, identify key factors, and propose "
        "the next experiment.\n\n"
        f"```json\n{json.dumps(comparison, indent=2)}\n```"
    )

    raw = run_json_agent(
        system=COMPARE_HYPOTHESIZER_PROMPT,
        user=user_message,
        default_model=COMPARE_MODEL_DEFAULT,
        client=client,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    return parse_json_response(
        raw,
        required_keys=("hypothesis", "key_factors", "next_experiment"),
        error_class=CompareAgentError,
    )
