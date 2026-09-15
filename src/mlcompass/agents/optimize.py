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

from ._common import (
    AgentResponseError,
    build_anthropic_agent,
    parse_json_response,
    resolve_llm_config,
    run_json_agent,
)

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
    model: str | None = None,
) -> Any:
    """Build the Anthropic optimize strategist agent (pure reasoner, no tools)."""
    resolved = resolve_llm_config(
        default_model=OPTIMIZE_MODEL_DEFAULT,
        provider="anthropic",
        model=model,
    )
    return build_anthropic_agent(
        system=OPTIMIZE_STRATEGIST_PROMPT,
        model=resolved.model,
        client=client,
        max_turns=2,
    )


def strategize_optimize(
    report: dict[str, Any],
    *,
    client: Any | None = None,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Ask a model to plan the next HPO sprint.

    Provider, model, base URL and API key each resolve in the same order:
    **explicit argument, then environment (``MLCOMPASS_LLM_PROVIDER``,
    ``MLCOMPASS_LLM_MODEL``, ``MLCOMPASS_LLM_BASE_URL``, and the provider's
    own key variable), then the built-in default.** With none of them set
    the strategist talks to Anthropic exactly as it always has.
    ``provider="openai"`` sends a chat completion instead, which reaches any
    OpenAI-compatible endpoint — including a local, keyless ollama server
    via ``base_url``.
    """
    payload = json.dumps(_compact(report), default=str)
    user_message = (
        "Here is the optimize report from mlcompass optimize. Please give "
        "a headline, the dominant pattern, and a 2-4 step plan.\n\n"
        f"```json\n{payload}\n```"
    )
    raw = run_json_agent(
        system=OPTIMIZE_STRATEGIST_PROMPT,
        user=user_message,
        default_model=OPTIMIZE_MODEL_DEFAULT,
        client=client,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
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
