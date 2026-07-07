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
#            input comes back we deterministically check THREE properties:
#
#            (1) Entity soundness — every cited column is in the evidence set.
#            (2) Value soundness  — every quantitative claim, emitted as a
#                structured {column, statistic, value} triple, matches the
#                value the deterministic layer actually measured (within
#                VALUE_TOLERANCE, so honest 2-decimal rounding passes).
#            (3) Completeness    — when the narrator commits to a verdict, it
#                must address the top-ranked candidate-leak column; silently
#                omitting the most critical evidence item is a violation.
#                Skipped when the verdict is "cannot_determine" (an explicit
#                abstention is not an omission).
#
#            On any violation we reject and retry with a corrective message;
#            after the retry budget is exhausted we strip residual unsound
#            entities/claims (omissions cannot be stripped — they are flagged).
#            So the worst-case guarantee — "every entity and every number the
#            user sees is present in, and equal to, the deterministic
#            evidence" — holds regardless of provider schema enforcement.
#
# Tier A is what positions the contract against the constrained-generation
# literature; Tier B is what makes the guarantee real, testable offline with a
# mock client, and independent of any single provider's schema enforcement.
# Together (1)+(2) give claim-level soundness and (3) gives critical-evidence
# completeness — both decidable because the evidence world is closed.

SUBMIT_TOOL_NAME = "submit_investigation"

# Absolute tolerance for verifying cited correlation values against the
# evidence. 0.005 means a value quoted to two decimal places (0.94 for a
# measured 0.9412) passes, while genuine misquotes (0.85 for 0.9412) fail.
VALUE_TOLERANCE = 0.005

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
2. Every correlation number you mention MUST also be reported in `claims` as
   {"column": ..., "statistic": "correlation", "value": ...}, copied EXACTLY
   from the evidence dictionary. Each claim is checked against the measured
   value; a mismatched number is rejected and you will be asked to answer
   again.
3. If you give a verdict other than "cannot_determine", you MUST address the
   first column in `candidate_leak_columns` (the top-ranked candidate) in
   `columns_referenced`. Silently skipping the strongest evidence is a
   contract violation.
4. If `candidate_leak_columns` is empty AND `perfect_match_rate` is below 0.95,
   set verdict to "cannot_determine" and leave `columns_referenced` empty
   rather than speculating about a leak you have no evidence for.
5. If `trustworthy_sample_size` is false, downgrade confidence to "low" or
   "cannot_determine" regardless of other signals.
6. `recommended_checks` must be MANUAL checks the user runs themselves
   (inspect file X, verify split Y, re-run on holdout Z) — NEVER code patches.
7. `narration` must paraphrase only facts present in the evidence dictionary.

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


def evidence_correlation_map(evidence: dict[str, Any]) -> dict[str, float]:
    """Map feature name -> measured correlation, from the evidence dict.

    This is the reference set Tier B verifies quantitative claims against:
    a cited correlation value is sound iff it matches this map within
    :data:`VALUE_TOLERANCE`.
    """
    out: dict[str, float] = {}
    for entry in evidence.get("target_feature_correlations") or []:
        if isinstance(entry, dict):
            feature = entry.get("feature")
            corr = entry.get("correlation")
            if feature is not None and isinstance(corr, (int, float)):
                out[str(feature)] = float(corr)
    return out


def top_candidate(evidence: dict[str, Any]) -> str | None:
    """The top-ranked candidate-leak column, or None if there is none.

    This is the completeness anchor: a committed verdict that does not address
    this column has omitted the most critical evidence item.
    """
    candidates = evidence.get("candidate_leak_columns") or []
    return str(candidates[0]) if candidates else None


_SUBMIT_DESCRIPTION = (
    "Submit the leakage investigation result. Cite only columns present in the "
    "evidence dictionary, and report every correlation number you mention as a "
    "structured claim copied exactly from the evidence."
)


def _submit_input_schema(
    allowed_columns: list[str], *, enforce_enum: bool = True, strict: bool = False
) -> dict[str, Any]:
    """The JSON schema for the submit tool, with the runtime enum domain.

    The ``columns_referenced`` item schema and the ``claims[].column`` field
    carry an ``enum`` equal to ``allowed_columns`` — the evidence-bound value
    domain. Verdict and confidence keep their static enums; only the column
    domain is computed at call time. This inner schema is provider-neutral; the
    per-provider wrappers below place it under ``input_schema`` (Anthropic) or
    ``parameters`` (OpenAI).

    ``enforce_enum=False`` drops the runtime column enums (Tier A) while
    keeping the field structure — used by the stress configuration of the
    measurement harness to demonstrate Tier B catching live violations on its
    own. Production callers leave it True.

    ``strict=True`` emits the provider-strict-compatible variant (every
    property required, ``additionalProperties: false`` on all objects) so the
    schema can be decode-enforced by providers that document strict tool use
    (H4 enforcement-dichotomy measurements). Semantics are unchanged; Tier B
    still never relies on provider enforcement.
    """
    col_schema: dict[str, Any] = {"type": "string"}
    if enforce_enum:
        col_schema = {"type": "string", "enum": list(allowed_columns)}
    claim_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "column": dict(col_schema),
            "statistic": {"type": "string", "enum": ["correlation"]},
            "value": {"type": "number"},
        },
        "required": ["column", "statistic", "value"],
    }
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": sorted(_VERDICT_VALUES)},
            "confidence": {"type": "string", "enum": sorted(_CONFIDENCE_VALUES)},
            "columns_referenced": {"type": "array", "items": dict(col_schema)},
            "claims": {"type": "array", "items": claim_schema},
            "narration": {"type": "string"},
            "recommended_checks": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["verdict", "confidence", "columns_referenced", "claims", "narration"],
    }
    if strict:
        schema["required"] = sorted(schema["properties"])
        schema["additionalProperties"] = False
        claim_schema["additionalProperties"] = False
    return schema


def build_submit_investigation_tool(
    allowed_columns: list[str], *, enforce_enum: bool = True, strict: bool = False
) -> dict[str, Any]:
    """Anthropic-format ``submit_investigation`` tool with the runtime enum.

    ``strict=True`` sets the Messages API's documented strict flag so the
    provider decode-enforces the schema (constrained decoding); the schema
    switches to its strict-compatible variant automatically.
    """
    tool: dict[str, Any] = {
        "name": SUBMIT_TOOL_NAME,
        "description": _SUBMIT_DESCRIPTION,
        "input_schema": _submit_input_schema(
            allowed_columns, enforce_enum=enforce_enum, strict=strict
        ),
    }
    if strict:
        tool["strict"] = True
    return tool


def build_submit_investigation_tool_openai(
    allowed_columns: list[str], *, enforce_enum: bool = True, strict: bool = False
) -> dict[str, Any]:
    """OpenAI-compatible (function) ``submit_investigation`` tool with the enum.

    Works with any OpenAI-compatible Chat Completions provider — including
    OpenAI-compatible endpoints such as DeepSeek (`base_url=
    https://api.deepseek.com`). The same inner schema is reused, so the runtime
    enum domain is identical across providers.

    ``strict=True`` sets the documented ``function.strict`` flag (decode-time
    schema enforcement on providers that support it) and switches to the
    strict-compatible schema variant.
    """
    function: dict[str, Any] = {
        "name": SUBMIT_TOOL_NAME,
        "description": _SUBMIT_DESCRIPTION,
        "parameters": _submit_input_schema(
            allowed_columns, enforce_enum=enforce_enum, strict=strict
        ),
    }
    if strict:
        function["strict"] = True
    return {"type": "function", "function": function}


def _extract_tool_input(response: Any, tool_name: str) -> dict[str, Any]:
    """Pull the input dict of the named tool_use block from a response."""
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
            raw = getattr(block, "input", {})
            return dict(raw) if isinstance(raw, dict) else {}
    return {}


def _emit_anthropic(
    api: Any,
    model: str,
    system: str,
    user: str,
    tool: dict[str, Any],
    temperature: float | None = None,
) -> dict[str, Any]:
    """One forced ``submit_investigation`` call against an Anthropic-style API."""
    extra: dict[str, Any] = {} if temperature is None else {"temperature": temperature}
    response = api.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        tools=[tool],
        tool_choice={"type": "tool", "name": SUBMIT_TOOL_NAME},
        messages=[{"role": "user", "content": user}],
        **extra,
    )
    return _extract_tool_input(response, SUBMIT_TOOL_NAME)


def _emit_openai(
    api: Any,
    model: str,
    system: str,
    user: str,
    tool: dict[str, Any],
    temperature: float | None = None,
) -> dict[str, Any]:
    """One forced ``submit_investigation`` call against an OpenAI-compatible API.

    Reads the tool-call arguments (a JSON string in the OpenAI schema) from
    ``choices[0].message.tool_calls[0]``. Returns an empty dict if the model
    declined to call the tool — which Tier B then treats as a non-fabrication.
    """
    extra: dict[str, Any] = {} if temperature is None else {"temperature": temperature}
    response = api.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tools=[tool],
        tool_choice="required",
        **extra,
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
    system_prompt: str | None = None,
    enforce_schema_enum: bool = True,
    strict_tools: bool = False,
    correction_style: str = "named",
    neutral_user: bool = False,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Narrate the evidence under the evidence-bound runtime-schema contract.

    Unlike :func:`investigate_leakage` (a prose narrator constrained only by
    its prompt), this path forces the answer through a ``submit_investigation``
    tool whose column enums are generated from ``evidence`` at call time
    (Tier A), then deterministically verifies three properties (Tier B):
    entity soundness (cited columns exist in the evidence), value soundness
    (every structured ``{column, statistic, value}`` claim matches the measured
    value within :data:`VALUE_TOLERANCE`), and completeness (a committed
    verdict addresses the top-ranked candidate column). Violations trigger a
    corrective retry; after the budget, unsound entities/claims are stripped
    and persistent omissions are flagged.

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
        system_prompt: Override for the system prompt. Defaults to the strict
            :data:`LEAKAGE_BOUND_PROMPT`. The measurement harness passes a bare
            prompt here in its stress configuration; production callers leave
            it None.
        enforce_schema_enum: When False, the Tier A runtime enums are dropped
            from the tool schema (field structure kept) so Tier B's catches are
            observable in isolation. Diagnostic use only; production callers
            leave it True.
        strict_tools: When True, the tool is emitted with the provider's
            documented strict flag and a strict-compatible schema variant, so
            capable providers decode-enforce it (H4 enforcement-dichotomy
            measurements). Tier B behaves identically either way.
        correction_style: ``"named"`` (default) sends the corrective retry
            with the offending items spelled out; ``"generic"`` sends a
            content-free rejection ("your previous answer was rejected") — the
            preregistered H5 control that separates the effect of NAMING the
            violation from the effect of merely getting a second attempt.
        neutral_user: When True, the user message is the bare evidence prompt
            with no mention of a contract — aligns stress-arm propensity with
            the bare-prompt baseline (analysis_plan A3.11). Production callers
            leave it False.
        temperature: Sampling temperature forwarded verbatim to the provider
            call, or None (default) to omit the parameter entirely and use
            the provider default. Measurement runs pin this so cross-provider
            rates are compared at a common decoding setting.

    Returns:
        Dict with the renderer-compatible keys (``verdict``, ``confidence``,
        ``evidence_cited``, ``primary_hypothesis``, ``recommended_checks``)
        plus contract telemetry: ``columns_referenced`` (the validated set),
        ``schema_rejections`` (how many times a violation was caught),
        ``rejection_kinds`` (per rejection round, which channels fired, e.g.
        ``["entity", "entity+value"]`` — violation-composition telemetry),
        ``attempts_made`` (provider calls issued, 1..max_retries+1),
        ``had_unrecoverable_violation`` (True if a phantom survived the retry
        budget and was stripped deterministically), and ``evidence_bound``.
    """
    allowed = evidence_allowed_columns(evidence)
    allowed_set = set(allowed)
    system = system_prompt if system_prompt is not None else LEAKAGE_BOUND_PROMPT
    if provider == "openai":
        tool = build_submit_investigation_tool_openai(
            allowed, enforce_enum=enforce_schema_enum, strict=strict_tools
        )
    else:
        tool = build_submit_investigation_tool(
            allowed, enforce_enum=enforce_schema_enum, strict=strict_tools
        )

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

    if neutral_user:
        base_user = (
            "Evidence dictionary:\n\n"
            f"{json.dumps(_compact(evidence), indent=2, default=str)}\n\n"
            "Investigate the suspicious metric."
        )
    else:
        base_user = (
            "Here is the leakage evidence from mlcompass.tools.leakage. Investigate "
            "it under the strict contract in your system prompt and submit your "
            "answer through the submit_investigation tool.\n\n"
            f"```json\n{json.dumps(_compact(evidence), default=str)}\n```"
        )

    corr_map = evidence_correlation_map(evidence)
    anchor = top_candidate(evidence)

    schema_rejections = 0
    rejection_kinds: list[str] = []
    attempts_made = 0
    correction = ""
    tool_input: dict[str, Any] = {}
    cited: list[str] = []
    claims: list[dict[str, Any]] = []
    omitted = False

    for attempt in range(max_retries + 1):
        attempts_made = attempt + 1
        user = base_user + correction
        if provider == "openai":
            tool_input = _emit_openai(api, model, system, user, tool, temperature=temperature)
        else:
            tool_input = _emit_anthropic(api, model, system, user, tool, temperature=temperature)
        cited = [str(c) for c in (tool_input.get("columns_referenced") or [])]
        claims = [c for c in (tool_input.get("claims") or []) if isinstance(c, dict)]

        # Tier B (1) — entity soundness.
        entity_violations = [c for c in cited if c not in allowed_set]
        # Tier B (2) — value soundness of structured claims.
        value_violations: list[str] = []
        for claim in claims:
            col = str(claim.get("column", ""))
            val = claim.get("value")
            if col not in corr_map:
                value_violations.append(f"{col}: not in evidence")
            elif (
                not isinstance(val, (int, float))
                or abs(float(val) - corr_map[col]) > VALUE_TOLERANCE
            ):
                value_violations.append(f"{col}: cited {val}, evidence says {corr_map[col]:.4f}")
        # Tier B (3) — completeness. Only when the narrator commits to a
        # verdict; an explicit abstention (or a declined call) is not an
        # omission. Claims columns count as addressing the anchor.
        raw_verdict = str(tool_input.get("verdict", "cannot_determine"))
        committed = (
            bool(tool_input)
            and raw_verdict in _VERDICT_VALUES
            and raw_verdict != "cannot_determine"
        )
        referenced = set(cited) | {str(c.get("column", "")) for c in claims}
        omitted = committed and anchor is not None and anchor not in referenced

        if not entity_violations and not value_violations and not omitted:
            break
        # Deterministic rejection, independent of the provider.
        schema_rejections += 1
        # Violation-composition telemetry (which channels fired this round).
        kinds: list[str] = []
        if entity_violations:
            kinds.append("entity")
        if value_violations:
            kinds.append("value")
        if omitted:
            kinds.append("omission")
        rejection_kinds.append("+".join(kinds))
        if attempt < max_retries:
            if correction_style == "generic":
                # H5 control (analysis_plan A3.2): a rejection carrying NO
                # information about what was wrong.
                correction = (
                    "\n\nYour previous answer was rejected. Answer again "
                    "through submit_investigation."
                )
                continue
            parts: list[str] = []
            if entity_violations:
                parts.append(
                    f"you cited columns NOT in the evidence dictionary: {entity_violations}; "
                    f"you may cite only these columns: {allowed}"
                )
            if value_violations:
                parts.append(
                    "these claims do not match the measured values: "
                    + "; ".join(value_violations)
                    + " — copy values exactly from the evidence"
                )
            if omitted:
                parts.append(
                    f"you committed to a verdict but did not address the top-ranked "
                    f"candidate-leak column '{anchor}' — address it or answer cannot_determine"
                )
            correction = (
                "\n\nYour previous answer violated the contract: "
                + ". Also, ".join(parts)
                + ". Re-answer through submit_investigation."
            )

    # Final deterministic strip — the worst-case guarantee. Any entity or claim
    # still unsound after the retry budget is removed before it reaches the
    # user. Omissions cannot be stripped; they are flagged instead.
    cited_clean = [c for c in cited if c in allowed_set]
    claims_clean = [
        c
        for c in claims
        if str(c.get("column", "")) in corr_map
        and isinstance(c.get("value"), (int, float))
        and abs(float(c["value"]) - corr_map[str(c["column"])]) <= VALUE_TOLERANCE
    ]
    had_unrecoverable = len(cited_clean) != len(cited) or len(claims_clean) != len(claims)

    verdict = str(tool_input.get("verdict", "cannot_determine"))
    if verdict not in _VERDICT_VALUES:
        verdict = "cannot_determine"
    confidence = str(tool_input.get("confidence", "cannot_determine"))
    if confidence not in _CONFIDENCE_VALUES:
        confidence = "cannot_determine"

    # Completeness is re-checked AFTER stripping: if the anchor was referenced
    # only through a claim that the strip removed, the returned response no
    # longer addresses it, and the omission flag must reflect that (an
    # omission cannot be repaired by deletion — Proposition 2).
    if not omitted and verdict != "cannot_determine" and anchor is not None:
        referenced_clean = set(cited_clean) | {str(c.get("column", "")) for c in claims_clean}
        omitted = bool(tool_input) and anchor not in referenced_clean

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
        "claims": claims_clean,
        "schema_rejections": schema_rejections,
        "rejection_kinds": rejection_kinds,
        "attempts_made": attempts_made,
        "had_unrecoverable_violation": had_unrecoverable,
        "omitted_critical_evidence": omitted,
        "evidence_bound": True,
    }


__all__ = [
    "LeakageAgentError",
    "build_leakage_agent",
    "investigate_leakage",
    "investigate_leakage_bound",
    "evidence_allowed_columns",
    "evidence_correlation_map",
    "top_candidate",
    "build_submit_investigation_tool",
    "build_submit_investigation_tool_openai",
    "SUBMIT_TOOL_NAME",
    "LEAKAGE_MODEL_DEFAULT",
    "VALUE_TOLERANCE",
]
