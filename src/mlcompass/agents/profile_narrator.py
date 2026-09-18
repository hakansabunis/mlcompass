"""A second evidence-closed narrator: dataset profiles, same contract.

The paper defines a *class* of tasks and measures one member of it. Two
independent reviewers made the same objection in the same words, and it is the
right objection: a class-level claim evaluated on a single evidence shape rests
on an argument rather than on evidence.

This is the second member. It narrates
:func:`mlcompass.tools.dataset.analyze_dataset` --- a deterministic profiler
that ships in the same tool --- under the same two-tier contract, through the
same verifier, with no branch anywhere that names either task. Everything
task-specific lives in
:data:`mlcompass.agents.evidence_contract.PROFILE`, which is 40 lines of
binder.

It is a genuinely different shape rather than leakage with different words:

* **Wide domain.** The entity set is *every* column in the frame, not a
  candidate list filtered by a correlation threshold. Leakage's $E$ carried
  ten; a profile carries as many as the data has, which is also what makes
  this the natural place to measure Tier A's cost as the enum grows.
* **Many statistics.** Thirteen quantities per column against leakage's one,
  and which ones exist depends on the column: a categorical column carries
  ``cardinality`` and no ``mean``. The statistic domain is therefore computed
  per call, not frozen in the schema.
* **The trap is the same and the bait is different.** Leakage's dominant
  failure is writing ``r2`` --- a statistic name sitting in $E$ --- into a
  column field. A profile puts thirteen statistic names beside N column names
  in one dictionary, so the same confusion has more ways to happen and nobody
  had to construct it.

What this module does NOT do is re-implement Tier B. If it did, the experiment
would measure two implementations rather than one contract over two tasks, and
would be worthless for the purpose. The verification, the stripping, the
corrective retry and the schema all come from ``evidence_contract``.
"""

from __future__ import annotations

import json
from typing import Any

from .evidence_contract import (
    PROFILE as PROFILE_CONTRACT,
)
from .evidence_contract import (
    build_payload_schema,
    correction_text,
    is_committed,
    strip_unsound,
    verify,
)

SUBMIT_TOOL_NAME = "submit_profile_assessment"

PROFILE_MODEL_DEFAULT = "claude-opus-4-7"

_SUBMIT_DESCRIPTION = (
    "Submit the dataset profile assessment. Cite only columns present in the "
    "profile, and report every number you mention as a structured claim "
    "{column, statistic, value} copied exactly from the profile."
)

#: The strict prompt the shipped path would use. Kept deliberately parallel to
#: the leakage contract's, so a difference in measured rates between the two
#: tasks is a difference in the tasks and not in how hard we asked.
PROFILE_BOUND_PROMPT = """You are mlcompass-profile-narrator.

A deterministic profiler has analysed a dataset and produced the evidence
dictionary below: per-column type, missing counts and rates, descriptive
statistics for numeric columns, cardinality for categorical ones, and outlier
counts.

Answer through the submit_profile_assessment tool. The contract:

1. Cite only column names that appear in the profile. Statistic names
   (mean, missing_pct, cardinality, ...) are NOT column names.
2. Every number you state must be a structured claim of the form
   {column, statistic, value}, copied exactly from the profile.
3. If you commit to a verdict, address the column carrying the most missing
   data. If you cannot assess the dataset, answer cannot_determine.
"""

#: The bare prompt: the same request with no faithfulness rule at all. This is
#: the floor arm, and the only difference from the prompt above is the contract.
PROFILE_BARE_PROMPT = """You are a data-quality assistant.

A profiler has analysed a dataset and produced the evidence dictionary below.
Assess whether the data is ready to train on, and answer through the
submit_profile_assessment tool: a verdict, the columns you referenced, any
numbers you mention as {column, statistic, value} claims, and a short
narration.
"""


def compact(evidence: dict[str, Any], *, max_columns: int | None = None) -> dict[str, Any]:
    """Trim the profile to what a narrator needs, keeping the shape intact.

    ``analyze_dataset`` carries per-column top-value lists and sample values
    that make the prompt large without changing what may be claimed. Dropping
    them keeps the token cost comparable to the leakage task, which matters
    because the two rates are compared.

    ``max_columns`` truncates the column list. That is the scalability knob:
    the same evidence at 10, 50, 100 and 500 columns gives four points on
    Tier~A's cost curve without changing anything else about the task.
    """
    columns = []
    for col in evidence.get("columns") or []:
        if not isinstance(col, dict):
            continue
        keep = {
            k: v
            for k, v in col.items()
            if k in ("name", "type", "missing_count", "missing_pct",
                     "cardinality", "zero_ratio", "stats", "outliers")
        }
        columns.append(keep)
        if max_columns is not None and len(columns) >= max_columns:
            break
    return {
        "shape": evidence.get("shape"),
        "columns": columns,
        "target_hint": evidence.get("target_hint"),
        "task_hint": evidence.get("task_hint"),
        "warnings": evidence.get("warnings"),
    }


def build_submit_tool(
    bound: Any, *, enforce_enum: bool = True, strict: bool = False
) -> dict[str, Any]:
    """Anthropic-format tool. The enums come from the bound evidence."""
    tool: dict[str, Any] = {
        "name": SUBMIT_TOOL_NAME,
        "description": _SUBMIT_DESCRIPTION,
        "input_schema": build_payload_schema(
            bound, PROFILE_CONTRACT, enforce_enum=enforce_enum, strict=strict
        ),
    }
    if strict:
        tool["strict"] = True
    return tool


def build_submit_tool_openai(
    bound: Any, *, enforce_enum: bool = True, strict: bool = False
) -> dict[str, Any]:
    """OpenAI-format wrapper around the identical inner schema."""
    fn: dict[str, Any] = {
        "name": SUBMIT_TOOL_NAME,
        "description": _SUBMIT_DESCRIPTION,
        "parameters": build_payload_schema(
            bound, PROFILE_CONTRACT, enforce_enum=enforce_enum, strict=strict
        ),
    }
    if strict:
        fn["strict"] = True
    return {"type": "function", "function": fn}


def _extract_tool_input(response: Any, tool_name: str) -> dict[str, Any]:
    """Pull the tool payload out of either provider's response shape."""
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", "") == tool_name:
            return dict(getattr(block, "input", {}) or {})
    for choice in getattr(response, "choices", None) or []:
        calls = getattr(getattr(choice, "message", None), "tool_calls", None) or []
        for call in calls:
            fn = getattr(call, "function", None)
            if fn is not None and getattr(fn, "name", "") == tool_name:
                try:
                    return dict(json.loads(getattr(fn, "arguments", "") or "{}"))
                except json.JSONDecodeError:
                    return {}
    return {}


def narrate_profile_bound(
    evidence: dict[str, Any],
    *,
    client: Any,
    model: str = PROFILE_MODEL_DEFAULT,
    provider: str = "openai",
    max_retries: int = 2,
    system_prompt: str | None = None,
    enforce_schema_enum: bool = True,
    strict_tools: bool = False,
    correction_style: str = "named",
    temperature: float | None = None,
    max_columns: int | None = None,
) -> dict[str, Any]:
    """Narrate a dataset profile under the evidence-bound runtime contract.

    The arguments mirror :func:`investigate_leakage_bound` exactly, including
    the diagnostic ones, so an arm defined for one task means the same thing on
    the other and the two batteries are comparable by construction.

    Returns the same telemetry keys the leakage path returns, plus
    ``columns_referenced``, ``claims`` and ``narration``.
    """
    trimmed = compact(evidence, max_columns=max_columns)
    bound = PROFILE_CONTRACT.bind(trimmed)
    system = system_prompt if system_prompt is not None else PROFILE_BOUND_PROMPT

    builder = build_submit_tool_openai if provider == "openai" else build_submit_tool
    tool = builder(bound, enforce_enum=enforce_schema_enum, strict=strict_tools)

    base_user = (
        "Dataset profile:\n\n"
        f"```json\n{json.dumps(trimmed, default=str)}\n```\n\n"
        f"Assess it and answer through {SUBMIT_TOOL_NAME}."
    )

    schema_rejections = 0
    rejection_kinds: list[str] = []
    attempts_made = 0
    correction = ""
    payload: dict[str, Any] = {}
    violations = verify({}, bound, PROFILE_CONTRACT)

    for attempt in range(max_retries + 1):
        attempts_made = attempt + 1
        user = base_user + correction
        if provider == "openai":
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "tools": [tool],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": SUBMIT_TOOL_NAME},
                },
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            response = client.chat.completions.create(**kwargs)
        else:
            kwargs = {
                "model": model,
                "max_tokens": 2048,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "tools": [tool],
                "tool_choice": {"type": "tool", "name": SUBMIT_TOOL_NAME},
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            response = client.messages.create(**kwargs)

        payload = _extract_tool_input(response, SUBMIT_TOOL_NAME)
        violations = verify(payload, bound, PROFILE_CONTRACT)
        if not violations:
            break

        schema_rejections += 1
        rejection_kinds.append(violations.kinds)
        if attempt < max_retries:
            if correction_style == "generic":
                correction = (
                    f"\n\nYour previous answer was rejected. Answer again "
                    f"through {SUBMIT_TOOL_NAME}."
                )
            else:
                correction = correction_text(
                    violations, bound, PROFILE_CONTRACT, tool_name=SUBMIT_TOOL_NAME
                )

    cited_clean, claims_clean = strip_unsound(payload, bound, PROFILE_CONTRACT)
    cited_raw = [str(c) for c in (payload.get("columns_referenced") or [])]
    claims_raw = [c for c in (payload.get("claims") or []) if isinstance(c, dict)]
    had_unrecoverable = (
        len(cited_clean) != len(cited_raw) or len(claims_clean) != len(claims_raw)
    )

    verdict = str(payload.get("verdict", PROFILE_CONTRACT.abstention))
    if verdict not in PROFILE_CONTRACT.verdict_values:
        verdict = PROFILE_CONTRACT.abstention
    confidence = str(payload.get("confidence", PROFILE_CONTRACT.abstention))
    if confidence not in PROFILE_CONTRACT.confidence_values:
        confidence = PROFILE_CONTRACT.abstention

    # Completeness is re-checked after stripping: an anchor referenced only
    # through a claim the strip removed is no longer addressed, and an omission
    # cannot be repaired by deletion.
    omitted = violations.omitted
    if not omitted and verdict != PROFILE_CONTRACT.abstention and bound.anchor is not None:
        referenced = set(cited_clean) | {
            str(c.get("column", "")) for c in claims_clean
        }
        omitted = bool(payload) and bound.anchor not in referenced

    return {
        "verdict": verdict,
        "confidence": confidence,
        "columns_referenced": cited_clean,
        "claims": claims_clean,
        "narration": str(payload.get("narration", "")).strip(),
        "recommended_checks": [
            str(c).strip() for c in (payload.get("recommended_checks") or []) if c
        ],
        "schema_rejections": schema_rejections,
        "rejection_kinds": rejection_kinds,
        "attempts_made": attempts_made,
        "had_unrecoverable_violation": had_unrecoverable,
        "omitted_critical_evidence": omitted,
        "critical_column": bound.anchor,
        "committed": is_committed(payload, PROFILE_CONTRACT),
        "evidence_bound": True,
        "admissible_columns": len(bound.domains["columns_referenced"]),
        "admissible_statistics": len(bound.domains["claims[].statistic"]),
    }


__all__ = [
    "PROFILE_BARE_PROMPT",
    "PROFILE_BOUND_PROMPT",
    "PROFILE_MODEL_DEFAULT",
    "SUBMIT_TOOL_NAME",
    "build_submit_tool",
    "build_submit_tool_openai",
    "compact",
    "narrate_profile_bound",
]
