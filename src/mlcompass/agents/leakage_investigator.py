"""LLM narrator over the deterministic leakage-evidence dict.

Strict anti-hallucination contract:

1. The agent receives the structured evidence from
   :func:`mlcompass.tools.leakage.detect_leakage` — and nothing else.
2. The system prompt tells it explicitly to **cite only evidence
   present in the input dict**. If a field is empty, the agent is
   instructed to say "no evidence found" rather than invent a cause.
3. The recommendations list is constrained to **manual checks** —
   we never let the LLM emit code patches. Code-level fixes belong
   to the user, not the advisor.

The output schema is intentionally narrow so the renderer in
``ui/evaluate.py`` can show it as a compact panel beside the
evaluation metrics — no rambling prose blocks.
"""

from __future__ import annotations

import json
from typing import Any

from agentlite import Agent

from ._common import AgentResponseError, parse_json_response

LEAKAGE_MODEL_DEFAULT = "claude-opus-4-7"

LEAKAGE_INVESTIGATOR_PROMPT = """You are mlcompass-leakage-investigator.

The mlcompass evaluator detected a suspiciously perfect metric on a
predictions table. A deterministic tool (mlcompass.tools.leakage) has
already gathered evidence: per-feature correlations against the
ground-truth column, the exact y_pred == y_true match rate, candidate
leak columns above the correlation threshold, and the sample size.

Your job is to read this evidence dict and write a short, grounded
investigation report. Reply with a single JSON object — no preamble,
no markdown fences — matching this shape:

{
  "verdict": "leakage_likely" | "leakage_uncertain" | "score_legitimate" | "cannot_determine",
  "confidence": "high" | "medium" | "low" | "cannot_determine",
  "evidence_cited": [
    "concrete fact pulled verbatim from the input dict, no embellishment",
    ...
  ],
  "primary_hypothesis": "one-sentence guess at the most likely cause, OR 'cannot determine without seeing the training script / dataset'",
  "recommended_checks": [
    "manual check the user should perform (max 25 words, no code patches)",
    ...
  ]
}

STRICT RULES — these guard against hallucination:

1. Cite ONLY items present in the evidence dict. Do NOT invent
   columns, features, or filenames that don't appear there.
2. If "candidate_leak_columns" is empty AND "perfect_match_rate" is
   below 0.95, say "cannot_determine". Do NOT speculate about
   train/test contamination or feature leakage you have no evidence
   for.
3. If "trustworthy_sample_size" is false, downgrade confidence to
   "low" or "cannot_determine" regardless of other signals.
4. Recommendations must be MANUAL checks (look at file X, verify
   split Y, rerun on holdout Z). NEVER propose code patches.
5. "evidence_cited" entries must be paraphrasable to facts in the
   dict — if you cannot point to a key in the evidence, leave the
   list empty rather than fill it with speculation.

The verdict mapping:
- "leakage_likely": ≥ 1 candidate_leak_columns OR perfect_match_rate ≥ 0.95.
- "leakage_uncertain": metric is suspiciously high but no clear smoking
   gun in the evidence.
- "score_legitimate": metric is suspicious but evidence suggests it's a
   genuinely strong model on small / easy data.
- "cannot_determine": insufficient evidence to call it either way."""


class LeakageAgentError(AgentResponseError):
    """Raised when the investigator returns malformed JSON."""


def build_leakage_agent(
    *,
    client: Any | None = None,
    model: str = LEAKAGE_MODEL_DEFAULT,
) -> Agent:
    """Build the leakage investigator agent (pure reasoner, no tools)."""
    return Agent(
        model=model,
        system=LEAKAGE_INVESTIGATOR_PROMPT,
        tools=[],
        client=client,
        max_turns=2,
    )


def investigate_leakage(
    evidence: dict[str, Any],
    *,
    client: Any | None = None,
    model: str = LEAKAGE_MODEL_DEFAULT,
) -> dict[str, Any]:
    """Narrate the deterministic leakage evidence dict.

    Returns:
        Dict with ``verdict``, ``confidence``, ``evidence_cited``,
        ``primary_hypothesis``, ``recommended_checks``.

    Raises:
        LeakageAgentError: On malformed agent output.
    """
    agent = build_leakage_agent(client=client, model=model)
    payload = json.dumps(_compact(evidence), default=str)
    user_message = (
        "Here is the leakage evidence from mlcompass.tools.leakage. Investigate "
        "it under the strict rules in your system prompt and reply with the "
        "required JSON.\n\n"
        f"```json\n{payload}\n```"
    )
    raw = agent.run(user_message)
    parsed = parse_json_response(
        raw,
        required_keys=(
            "verdict",
            "confidence",
            "evidence_cited",
            "primary_hypothesis",
            "recommended_checks",
        ),
        error_class=LeakageAgentError,
    )

    # Clamp the output to the expected enum values so downstream
    # rendering stays stable even if the LLM hallucinates a verdict.
    verdict = str(parsed.get("verdict", "cannot_determine"))
    if verdict not in _VERDICT_VALUES:
        verdict = "cannot_determine"
    confidence = str(parsed.get("confidence", "cannot_determine"))
    if confidence not in _CONFIDENCE_VALUES:
        confidence = "cannot_determine"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "evidence_cited": [str(e).strip() for e in (parsed.get("evidence_cited") or []) if e],
        "primary_hypothesis": str(parsed.get("primary_hypothesis", "")).strip(),
        "recommended_checks": [
            str(c).strip() for c in (parsed.get("recommended_checks") or []) if c
        ],
    }


_VERDICT_VALUES = frozenset(
    {"leakage_likely", "leakage_uncertain", "score_legitimate", "cannot_determine"}
)
_CONFIDENCE_VALUES = frozenset({"high", "medium", "low", "cannot_determine"})


def _compact(evidence: dict[str, Any]) -> dict[str, Any]:
    """Drop fields the agent doesn't need — keep the prompt small."""
    return {
        "row_count": evidence.get("row_count"),
        "trustworthy_sample_size": evidence.get("trustworthy_sample_size"),
        "suspicious_metric": evidence.get("suspicious_metric"),
        "target_feature_correlations": evidence.get("target_feature_correlations"),
        "perfect_match_rate": evidence.get("perfect_match_rate"),
        "candidate_leak_columns": evidence.get("candidate_leak_columns"),
        "notes": evidence.get("notes"),
    }
