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


# =========================================================================== #
# Evidence-bound runtime-schema contract (the schema-enforced narrator)       #
# =========================================================================== #
#
# The prose narrator above (``investigate_leakage``) relies entirely on the
# system prompt to keep the model from citing columns that are not in the
# evidence. Prompts are advisory: a strict prompt lowers, but does not
# eliminate, phantom-column fabrication. This section adds the enforced
# path used in production, in two tiers:
#
#   Tier A — Runtime schema enum (constrained generation). The narrator must
#            answer through a ``submit_investigation`` tool whose
#            ``columns_referenced`` field is restricted, AT CALL TIME, by a
#            JSON-schema ``enum`` generated from the evidence dictionary. The
#            enum domain is exactly the set of columns the deterministic layer
#            observed — not a static type.
#
#   Tier B — Deterministic post-validation (provider-independent guarantee).
#            We do not *trust* the provider to honour the enum. After the tool
#            input comes back, we re-check every cited column against the
#            evidence set ourselves. On a violation we reject and retry with a
#            corrective message; after the retry budget is exhausted we strip
#            any residual out-of-evidence column. So the worst-case guarantee
#            ("every column the user sees is in the evidence") holds regardless
#            of whether the provider enforced the enum.
#
# Tier A is what positions the contract against the constrained-generation
# literature; Tier B is what makes the guarantee real, testable offline with a
# mock client, and independent of any single provider's schema enforcement.

SUBMIT_TOOL_NAME = "submit_investigation"

LEAKAGE_BOUND_PROMPT = """You are mlcompass-leakage-investigator.

A deterministic tool (mlcompass.tools.leakage) detected a suspiciously
perfect metric on a predictions table and gathered evidence: per-feature
correlations with the ground-truth column, the exact y_pred == y_true match
rate, candidate leak columns above the correlation threshold, and the sample
size.

Call the `submit_investigation` tool exactly once with your findings.

STRICT CONTRACT — these guard against hallucination:

1. In `columns_referenced`, list ONLY the columns you actually cite, and cite
   ONLY columns that appear in the evidence dictionary. The tool input schema
   restricts this field to the evidence columns; naming any other column is
   rejected and you will be asked to answer again.
2. If `candidate_leak_columns` is empty AND `perfect_match_rate` is below 0.95,
   set verdict to "cannot_determine" and leave `columns_referenced` empty
   rather than speculating about a leak you have no evidence for.
3. If `trustworthy_sample_size` is false, downgrade confidence to "low" or
   "cannot_determine" regardless of other signals.
4. `recommended_checks` must be MANUAL checks the user runs themselves
   (inspect file X, verify split Y, re-run on holdout Z) — NEVER code patches.
5. `narration` must paraphrase only facts present in the evidence dictionary.

Verdict mapping:
- "leakage_likely": >= 1 candidate_leak_columns OR perfect_match_rate >= 0.95.
- "leakage_uncertain": metric is suspiciously high but no clear smoking gun.
- "score_legitimate": metric is suspicious but evidence suggests a genuinely
   strong model on small / easy data.
- "cannot_determine": insufficient evidence to call it either way."""


def evidence_allowed_columns(evidence: dict[str, Any]) -> list[str]:
    """The runtime enum domain: every column the deterministic layer observed.

    This is the call-time value domain bound into the tool schema. It is the
    union of the candidate leak columns and every feature the correlation pass
    reported, so the domain is well defined regardless of the correlation
    threshold used upstream.

    Returns a sorted list for deterministic schema generation (stable enum
    order across runs makes the contract reproducible and testable).
    """
    allowed: set[str] = set()
    for col in evidence.get("candidate_leak_columns") or []:
        if col is not None:
            allowed.add(str(col))
    for entry in evidence.get("target_feature_correlations") or []:
        if isinstance(entry, dict):
            feature = entry.get("feature")
            if feature is not None:
                allowed.add(str(feature))
    return sorted(allowed)


_SUBMIT_DESCRIPTION = (
    "Submit the leakage investigation result. Cite only columns present in the "
    "evidence dictionary."
)


def _submit_input_schema(allowed_columns: list[str]) -> dict[str, Any]:
    """The JSON schema for the submit tool, with the runtime enum domain.

    The ``columns_referenced`` item schema carries an ``enum`` equal to
    ``allowed_columns`` — the evidence-bound value domain. Verdict and
    confidence keep their static enums; only the column domain is computed at
    call time. This inner schema is provider-neutral; the per-provider wrappers
    below place it under ``input_schema`` (Anthropic) or ``parameters`` (OpenAI).
    """
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": sorted(_VERDICT_VALUES)},
            "confidence": {"type": "string", "enum": sorted(_CONFIDENCE_VALUES)},
            "columns_referenced": {
                "type": "array",
                "items": {"type": "string", "enum": list(allowed_columns)},
            },
            "narration": {"type": "string"},
            "recommended_checks": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["verdict", "confidence", "columns_referenced", "narration"],
    }


def build_submit_investigation_tool(allowed_columns: list[str]) -> dict[str, Any]:
    """Anthropic-format ``submit_investigation`` tool with the runtime enum."""
    return {
        "name": SUBMIT_TOOL_NAME,
        "description": _SUBMIT_DESCRIPTION,
        "input_schema": _submit_input_schema(allowed_columns),
    }


def build_submit_investigation_tool_openai(allowed_columns: list[str]) -> dict[str, Any]:
    """OpenAI-compatible (function) ``submit_investigation`` tool with the enum.

    Works with any OpenAI-compatible Chat Completions provider — including
    OpenAI-compatible endpoints such as DeepSeek (`base_url=
    https://api.deepseek.com`). The same inner schema is reused, so the runtime
    enum domain is identical across providers.
    """
    return {
        "type": "function",
        "function": {
            "name": SUBMIT_TOOL_NAME,
            "description": _SUBMIT_DESCRIPTION,
            "parameters": _submit_input_schema(allowed_columns),
        },
    }


def _extract_tool_input(response: Any, tool_name: str) -> dict[str, Any]:
    """Pull the input dict of the named tool_use block from a response."""
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
            raw = getattr(block, "input", {})
            return dict(raw) if isinstance(raw, dict) else {}
    return {}


def _emit_anthropic(api: Any, model: str, user: str, allowed: list[str]) -> dict[str, Any]:
    """One forced ``submit_investigation`` call against an Anthropic-style API."""
    response = api.messages.create(
        model=model,
        max_tokens=1024,
        system=LEAKAGE_BOUND_PROMPT,
        tools=[build_submit_investigation_tool(allowed)],
        tool_choice={"type": "tool", "name": SUBMIT_TOOL_NAME},
        messages=[{"role": "user", "content": user}],
    )
    return _extract_tool_input(response, SUBMIT_TOOL_NAME)


def _emit_openai(api: Any, model: str, user: str, allowed: list[str]) -> dict[str, Any]:
    """One forced ``submit_investigation`` call against an OpenAI-compatible API.

    Reads the tool-call arguments (a JSON string in the OpenAI schema) from
    ``choices[0].message.tool_calls[0]``. Returns an empty dict if the model
    declined to call the tool — which Tier B then treats as a non-fabrication.
    """
    response = api.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LEAKAGE_BOUND_PROMPT},
            {"role": "user", "content": user},
        ],
        tools=[build_submit_investigation_tool_openai(allowed)],
        tool_choice="required",
    )
    message = response.choices[0].message
    calls = getattr(message, "tool_calls", None) or []
    if not calls:
        return {}
    args = calls[0].function.arguments
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
    except (json.JSONDecodeError, TypeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def investigate_leakage_bound(
    evidence: dict[str, Any],
    *,
    client: Any | None = None,
    model: str = LEAKAGE_MODEL_DEFAULT,
    max_retries: int = 2,
    provider: str = "anthropic",
) -> dict[str, Any]:
    """Narrate the evidence under the evidence-bound runtime-schema contract.

    Unlike :func:`investigate_leakage` (a prose narrator constrained only by
    its prompt), this path forces the answer through a ``submit_investigation``
    tool whose ``columns_referenced`` enum is generated from ``evidence`` at
    call time (Tier A), and then deterministically re-validates the cited
    columns against the evidence set, retrying on a violation and stripping any
    residual out-of-evidence column (Tier B).

    Args:
        evidence: The dict from :func:`mlcompass.tools.leakage.detect_leakage`.
        client: Provider client (injected in tests). Falls back to a real
            client only if ``None``: an Anthropic client for ``provider=
            "anthropic"`` or an OpenAI client for ``provider="openai"``.
        model: Model identifier passed to the provider call.
        max_retries: How many corrective retries to issue on a violation
            before falling back to deterministic stripping.
        provider: ``"anthropic"`` (default, Messages API) or ``"openai"`` (Chat
            Completions API, also used for OpenAI-compatible endpoints such as
            DeepSeek). Tier B (deterministic validation) is identical for both —
            that is what makes the worst-case guarantee provider-independent.

    Returns:
        Dict with the renderer-compatible keys (``verdict``, ``confidence``,
        ``evidence_cited``, ``primary_hypothesis``, ``recommended_checks``)
        plus contract telemetry: ``columns_referenced`` (the validated set),
        ``schema_rejections`` (how many times a violation was caught),
        ``had_unrecoverable_violation`` (True if a phantom survived the retry
        budget and was stripped deterministically), and ``evidence_bound``.
    """
    allowed = evidence_allowed_columns(evidence)
    allowed_set = set(allowed)

    # Keep the client untyped (Any) — the same discipline the agentlite path
    # uses. We pass raw dict tool/tool_choice params, which the provider's
    # strongly-typed overloads would otherwise reject under mypy --strict.
    api: Any = client
    if api is None:
        if provider == "openai":
            from openai import OpenAI

            api = OpenAI()
        else:
            import anthropic

            api = anthropic.Anthropic()

    base_user = (
        "Here is the leakage evidence from mlcompass.tools.leakage. Investigate "
        "it under the strict contract in your system prompt and submit your "
        "answer through the submit_investigation tool.\n\n"
        f"```json\n{json.dumps(_compact(evidence), default=str)}\n```"
    )

    schema_rejections = 0
    correction = ""
    tool_input: dict[str, Any] = {}
    cited: list[str] = []

    for attempt in range(max_retries + 1):
        user = base_user + correction
        if provider == "openai":
            tool_input = _emit_openai(api, model, user, allowed)
        else:
            tool_input = _emit_anthropic(api, model, user, allowed)
        cited = [str(c) for c in (tool_input.get("columns_referenced") or [])]
        violations = [c for c in cited if c not in allowed_set]
        if not violations:
            break
        # Tier B — deterministic rejection, independent of the provider.
        schema_rejections += 1
        if attempt < max_retries:
            correction = (
                "\n\nYour previous answer cited columns that are NOT in the "
                f"evidence dictionary: {violations}. You may cite only these "
                f"columns: {allowed}. Re-answer through submit_investigation, "
                "using columns_referenced drawn solely from that set."
            )

    # Final deterministic strip — the worst-case guarantee. Anything still out
    # of evidence after the retry budget is removed before it reaches the user.
    cited_clean = [c for c in cited if c in allowed_set]
    had_unrecoverable = len(cited_clean) != len(cited)

    verdict = str(tool_input.get("verdict", "cannot_determine"))
    if verdict not in _VERDICT_VALUES:
        verdict = "cannot_determine"
    confidence = str(tool_input.get("confidence", "cannot_determine"))
    if confidence not in _CONFIDENCE_VALUES:
        confidence = "cannot_determine"

    return {
        "verdict": verdict,
        "confidence": confidence,
        # Renderer-compatible keys (same shape as the prose path):
        "evidence_cited": cited_clean,
        "primary_hypothesis": str(tool_input.get("narration", "")).strip(),
        "recommended_checks": [
            str(c).strip() for c in (tool_input.get("recommended_checks") or []) if c
        ],
        # Contract telemetry:
        "columns_referenced": cited_clean,
        "schema_rejections": schema_rejections,
        "had_unrecoverable_violation": had_unrecoverable,
        "evidence_bound": True,
    }


__all__ = [
    "LeakageAgentError",
    "build_leakage_agent",
    "investigate_leakage",
    "investigate_leakage_bound",
    "evidence_allowed_columns",
    "build_submit_investigation_tool",
    "build_submit_investigation_tool_openai",
    "SUBMIT_TOOL_NAME",
    "LEAKAGE_MODEL_DEFAULT",
]
