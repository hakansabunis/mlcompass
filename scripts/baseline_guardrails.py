"""Guardrails AI baseline arms for the ablation harness (analysis_plan.md §6).

The reviewer-facing question this module answers is "how is the evidence-bound
contract different from Guardrails with a custom validator?". To answer it the
comparison has to be a comparison of MECHANISMS, so everything except the
enforcement mechanism is held fixed: same task, same evidence dict, same
model, same provider call, same forced ``submit_investigation`` tool with the
same OPEN (enum-free) schema the L1/L2 arms use, same sampling pin, and the
same three-channel scoring by the independent scorer.

Three arms are provided, matching the preregistered fairness design:

``guardrails_stock``
    ``Guard.for_pydantic(Investigation)`` with NO faithfulness validator.
    Guardrails still enforces the output STRUCTURE (JSON parse, field
    presence, field types) and reasks on a structural failure. This is what
    the toolkit gives you out of the box, and it is the honest answer to
    "does a validate-and-reask toolkit stop phantom columns by itself?".

``guardrails_choices``
    The same Guard plus ONE generic acceptable-values check on
    ``columns_referenced``, populated from the evidence at call time. This is
    the arm a reviewer means when they ask whether 35.0 % is a strawman: it is
    the competent Guardrails user who took the toolkit's own choice validator
    and fed it the list. No value tolerance, no anchor requirement, no bespoke
    logic -- strictly weaker than Tier B, by construction.

    Note for anyone checking the claim that the toolkit "ships" such a
    validator: it does not. A fresh ``pip install guardrails-ai==0.11.0``
    registers ZERO validators -- ``validators_registry`` is empty and
    ``guardrails.hub`` exports nothing. Each one is a separate
    ``guardrails hub install`` that fetches a manifest from
    hub.api.guardrailsai.com. The arm therefore reimplements the documented
    semantics locally, which keeps it runnable from the replication package
    alone and keeps the check short enough to audit by reading it.

``guardrails_tierb``
    The same Guard plus two custom validators that encode the Tier B checks
    (entity soundness, value soundness, completeness) with
    ``OnFailAction.REASK``. The validator is OURS; the loop is THEIRS. Any
    outcome difference against the shipped contract is therefore attributable
    to the loop and to what each mechanism does with a violation that
    survives the retry budget — not to one side having a weaker checker.
    This is deliberately the strongest configuration the toolkit reasonably
    allows: a strawman baseline would be worse than no baseline.

Configuration decisions, all load-bearing, all deliberate
---------------------------------------------------------

1. **Retry budget parity.** ``num_reasks=2`` mirrors the shipped contract's
   ``max_retries=2``, so both mechanisms get at most three provider calls.

2. **Field-scoped validators, not a whole-object validator.** Guardrails
   0.11.0 cannot reask on a whole-output validator: ``Guard.use(v,
   on="output")`` (and ``on="$"``) raises ``TypeError: 'NoneType' object is
   not iterable`` in ``guardrails/actions/reask.py`` when it builds the reask
   message, because a non-field ReAsk carries ``path=None``. Each check is
   therefore attached to the field whose value has to change to fix it.
   That is also what makes the loop work: Guardrails decides a failure was
   resolved by observing that the failing FIELD changed, so a validator
   attached to a field it does not own reports "unresolved" even after the
   model fixes the answer, and the call returns no output at all.

3. **Completeness is enforced unconditionally.** Tier B only requires the
   top-ranked candidate column when the narrator COMMITS to a verdict
   (an explicit ``cannot_determine`` is not an omission). A Guardrails
   validator on ``$.columns_referenced`` cannot see ``$.verdict`` — validators
   receive only their own field's value plus the call metadata — so the
   anchor requirement is unconditional here. This makes the baseline
   STRICTER than Tier B on this channel, never weaker; the cost shows up as
   extra reasks, which are recorded per response.

4. **Sampling is the harness's, not Guardrails'.** Guardrails passes its own
   ``temperature`` in the callable kwargs. The transport ignores those kwargs
   and uses the harness's pin (A3 implementation addendum), so this arm is
   sampled identically to every other arm.

5. **First attempt is byte-identical to L1.** Guardrails passes the caller's
   ``messages`` through verbatim on the first iteration and only injects its
   own schema/reask scaffolding on a reask. So attempt 1 of a Guardrails arm
   is the same provider call as an L1 sample, and any divergence is the loop.

6. **Telemetry is exhaustive.** ``validated_output is None`` after the retry
   budget means the toolkit withheld the answer — the mechanism difference
   against Tier B, which strips and still answers. Because that difference is
   the whole point, the final raw (unvalidated) payload is also recorded
   under ``baseline.passthrough`` so the "application ships the raw output
   anyway" variant can be scored later from the same logs, without new spend.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from typing import Any

# Transport contract: given the message list Guardrails wants sent, issue ONE
# provider call and return (tool_input_dict, usage_in, usage_out). The harness
# owns provider dispatch; this module owns Guardrails.
Transport = Callable[[list[dict[str, Any]]], "tuple[dict[str, Any], int | None, int | None]"]

GUARDRAILS_PACKAGE = "guardrails-ai"

# The two arms this module serves. Registered here so the harness and the
# tests agree on the names without importing guardrails.
GUARDRAILS_ARMS = ("guardrails_stock", "guardrails_choices", "guardrails_tierb")

# What each arm actually attaches, recorded in every response's telemetry so a
# reader never has to infer the configuration from the arm's name.
_VALIDATOR_NAMES: dict[str, list[str]] = {
    "guardrails_stock": [],
    "guardrails_choices": ["mlcompass/acceptable-values"],
    "guardrails_tierb": [
        "mlcompass/evidence-bound-columns",
        "mlcompass/evidence-bound-claims",
    ],
}

_VALIDATORS_REGISTERED = False


def guardrails_version() -> str | None:
    """Installed guardrails-ai version, or None when it is not installed."""
    try:
        import importlib.metadata as md

        return md.version(GUARDRAILS_PACKAGE)
    except Exception:  # noqa: BLE001 - absent package, broken metadata
        return None


def guardrails_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("guardrails") is not None


def _require() -> None:
    if not guardrails_available():
        raise SystemExit(
            "The Guardrails baseline arms need the toolkit installed:\n"
            "    pip install 'guardrails-ai==0.11.0'\n"
            "(declared as the 'baselines' extra in pyproject.toml)."
        )


# --------------------------------------------------------------------------- #
# Output model — mirrors the harness's OPEN schema exactly                     #
# --------------------------------------------------------------------------- #


def _build_models() -> tuple[Any, Any]:
    from pydantic import BaseModel, Field

    class Claim(BaseModel):
        column: str
        statistic: str
        value: float

    class Investigation(BaseModel):
        """Same fields, same required set as ``_OPEN_SCHEMA`` in the harness."""

        verdict: str
        confidence: str = Field(default="medium")
        columns_referenced: list[str]
        claims: list[Claim]
        narration: str

    return Claim, Investigation


# --------------------------------------------------------------------------- #
# The custom validators — Tier B's checks, expressed field by field            #
# --------------------------------------------------------------------------- #


def _register_validators() -> None:
    """Register the two Tier-B-equivalent validators exactly once.

    Both read their reference data (allowed columns / correlation map /
    anchor / tolerance) from the per-call ``metadata`` dict, so one
    registration serves every task and every evidence instance.
    """
    global _VALIDATORS_REGISTERED
    if _VALIDATORS_REGISTERED:
        return

    from guardrails.classes.validation.validation_result import (
        FailResult,
        PassResult,
        ValidationResult,
    )
    from guardrails.validator_base import Validator, register_validator

    @register_validator(name="mlcompass/evidence-bound-columns", data_type="list")
    class EvidenceBoundColumns(Validator):  # type: ignore[misc]
        """Tier B (1) entity soundness + Tier B (3) completeness.

        Both are repaired by editing ``columns_referenced``, which is why they
        live together on that field (see module docstring, decision 2).
        """

        def _validate(self, value: Any, metadata: dict[str, Any]) -> ValidationResult:
            allowed = [str(c) for c in (metadata.get("allowed_columns") or [])]
            anchor = metadata.get("anchor")
            cited = [str(c) for c in (value or [])]

            problems: list[str] = []
            off_evidence = [c for c in cited if c not in set(allowed)]
            if off_evidence:
                problems.append(
                    f"it cites columns that are NOT in the evidence dictionary: "
                    f"{off_evidence}. You may cite only these columns: {allowed}"
                )
            if anchor is not None and str(anchor) not in cited:
                problems.append(
                    f"it does not address '{anchor}', the top-ranked "
                    f"candidate-leak column, which the answer must address"
                )
            if problems:
                return FailResult(
                    error_message="columns_referenced is invalid because "
                    + "; also because ".join(problems)
                )
            return PassResult()

    @register_validator(name="mlcompass/evidence-bound-claims", data_type="list")
    class EvidenceBoundClaims(Validator):  # type: ignore[misc]
        """Tier B (2) value soundness of the structured claims."""

        def _validate(self, value: Any, metadata: dict[str, Any]) -> ValidationResult:
            corr_map = {str(k): float(v) for k, v in (metadata.get("corr_map") or {}).items()}
            tolerance = float(metadata.get("tolerance", 0.005))

            problems: list[str] = []
            for claim in value or []:
                claim = _as_dict(claim)
                if not claim:
                    continue
                column = str(claim.get("column", ""))
                cited_value = claim.get("value")
                if column not in corr_map:
                    problems.append(f"{column}: not a column in the evidence dictionary")
                elif (
                    not isinstance(cited_value, (int, float))
                    or abs(float(cited_value) - corr_map[column]) > tolerance
                ):
                    problems.append(
                        f"{column}: you wrote {cited_value}, the evidence "
                        f"says {corr_map[column]:.4f}"
                    )
            if problems:
                return FailResult(
                    error_message=(
                        "claims do not match the measured evidence: "
                        + "; ".join(problems)
                        + ". Copy every value exactly from the evidence dictionary."
                    )
                )
            return PassResult()

    @register_validator(name="mlcompass/acceptable-values", data_type="list")
    class AcceptableValues(Validator):  # type: ignore[misc]
        """The toolkit's generic acceptable-values check, given the evidence.

        This is the `guardrails_choices` arm and it exists to answer one
        objection: that 35.0 % measures a strawman, because a competent user
        would have taken Guardrails' own choice validator and populated it from
        the evidence. This arm IS that user.

        It is deliberately the whole of that configuration and nothing more.
        No value tolerance, no anchor requirement, no per-claim checking -- one
        membership test against a list supplied at call time, which is the
        documented semantics of the Hub's `ValidChoices`. It is therefore
        strictly weaker than `EvidenceBoundColumns` above, by construction, and
        any difference between the two arms is the difference between a generic
        check and a contract.

        Why it is written here rather than installed. A fresh
        `pip install guardrails-ai==0.11.0` registers ZERO validators:
        `validators_registry` is empty and `guardrails.hub` exports nothing.
        Every validator, this one included, is a separate `guardrails hub
        install` that fetches a manifest from hub.api.guardrailsai.com. The
        toolkit does not ship it. Reimplementing the documented semantics keeps
        the arm runnable from the replication package alone, and the semantics
        are simple enough that the reimplementation is checkable by reading it.
        """

        def _validate(self, value: Any, metadata: dict[str, Any]) -> ValidationResult:
            allowed = {str(c) for c in (metadata.get("allowed_columns") or [])}
            cited = [str(c) for c in (value or [])]
            off = [c for c in cited if c not in allowed]
            if off:
                return FailResult(
                    error_message=(
                        f"columns_referenced contains values that are not in the "
                        f"list of acceptable values: {off}. "
                        f"Acceptable values: {sorted(allowed)}"
                    )
                )
            return PassResult()

    _VALIDATORS_REGISTERED = True


def _silence_hub_telemetry() -> None:
    """Stop Guardrails exporting spans to its hub during a measurement run.

    ``settings.disable_tracing`` covers the Guard's own tracer and does not
    cover the hub tracer, which is a separate singleton that decides once, at
    ``initialize_tracer`` time, from ``settings.rc.enable_metrics``. Left
    alone it posts a span batch per call to an AWS endpoint and, when that
    endpoint cannot be resolved, retries with backoff before giving up. On a
    200-response arm that is real wall-clock on every response and a stream of
    stack traces over the run log.

    Two reasons this is not merely tidiness. A measurement run must not phone
    home: what it would be reporting is our own experiment's call pattern. And
    the retry delay lands inside the interval being timed, so an un-silenced
    run does not measure what it claims to.

    Best effort by construction. The internals it reaches into are private and
    may move between versions, so every step is guarded and a failure to
    silence is never a failure to run.
    """
    try:
        from guardrails import settings

        rc = getattr(settings, "rc", None)
        if rc is not None and hasattr(rc, "enable_metrics"):
            rc.enable_metrics = False
    except Exception:  # noqa: BLE001 - telemetry is never worth a crash
        pass

    try:
        from guardrails.hub_telemetry.hub_tracing import HubTelemetry

        # The singleton may already have been constructed at import, in which
        # case the constructor's reading of enable_metrics is already stale.
        instance = getattr(HubTelemetry, "_instance", None)
        for target in (instance, HubTelemetry):
            if target is not None:
                with contextlib.suppress(Exception):
                    target._enabled = False  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        result = dump()
        return result if isinstance(result, dict) else {}
    return {}


# --------------------------------------------------------------------------- #
# Guard construction                                                           #
# --------------------------------------------------------------------------- #


def build_guard(*, arm: str) -> Any:
    """Build the Guard for one arm.

    ``guardrails_stock``
        Structural validation and reask only. What the toolkit gives you out
        of the box, which is to say: no faithfulness checking at all.
    ``guardrails_choices``
        One generic acceptable-values check on ``columns_referenced``,
        populated from the evidence at call time.
    ``guardrails_tierb``
        The Tier-B-equivalent validators: entity, value and completeness.
    """
    _require()
    from guardrails import settings

    # No telemetry export: measurement runs must not phone home, and the
    # exporter otherwise retries against an unreachable host on every call.
    settings.disable_tracing = True
    _silence_hub_telemetry()

    from guardrails import Guard, OnFailAction

    if arm not in GUARDRAILS_ARMS:
        raise ValueError(f"unknown Guardrails arm {arm!r}; expected one of {GUARDRAILS_ARMS}")

    _, investigation = _build_models()
    guard = Guard.for_pydantic(investigation)
    if arm == "guardrails_stock":
        return guard

    _register_validators()
    from guardrails.validator_base import validators_registry

    if arm == "guardrails_choices":
        choices_validator = validators_registry["mlcompass/acceptable-values"]
        return guard.use(choices_validator(on_fail=OnFailAction.REASK), on="$.columns_referenced")

    columns_validator = validators_registry["mlcompass/evidence-bound-columns"]
    claims_validator = validators_registry["mlcompass/evidence-bound-claims"]
    return guard.use(columns_validator(on_fail=OnFailAction.REASK), on="$.columns_referenced").use(
        claims_validator(on_fail=OnFailAction.REASK), on="$.claims"
    )


# --------------------------------------------------------------------------- #
# One sampled response through the Guardrails loop                             #
# --------------------------------------------------------------------------- #


def run_guardrails_once(
    transport: Transport,
    *,
    system_prompt: str,
    user_message: str,
    allowed_columns: list[str],
    corr_map: dict[str, float],
    anchor: str | None,
    tolerance: float,
    arm: str,
    num_reasks: int = 2,
) -> dict[str, Any]:
    """Sample ONE response through Guardrails' validate-and-reask loop.

    Returns the user-facing result plus baseline telemetry. Scoring fields
    (``columns`` / ``claims`` / ``verdict``) carry what Guardrails actually
    handed back: when validation never passes within the retry budget the
    toolkit returns no validated output, and those fields are empty. The
    unvalidated final payload is preserved under ``baseline.passthrough`` so
    the alternative "ship it anyway" reading is recoverable from the same log.
    """
    guard = build_guard(arm=arm)

    calls: dict[str, Any] = {"n": 0, "usage_in": 0, "usage_out": 0, "usage_seen": False}
    last_payload: dict[str, Any] = {}

    def llm_callable(*args: Any, messages: Any = None, **kwargs: Any) -> str:
        """Guardrails' LLM seam. ``kwargs`` (Guardrails' own temperature and
        friends) are deliberately ignored: the harness owns sampling."""
        nonlocal last_payload
        tool_input, usage_in, usage_out = transport(list(messages or []))
        calls["n"] += 1
        if usage_in is not None:
            calls["usage_in"] += int(usage_in)
            calls["usage_seen"] = True
        if usage_out is not None:
            calls["usage_out"] += int(usage_out)
            calls["usage_seen"] = True
        last_payload = tool_input if isinstance(tool_input, dict) else {}
        return json.dumps(last_payload, default=str)

    outcome = guard(
        llm_callable,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        num_reasks=num_reasks,
        metadata={
            "allowed_columns": list(allowed_columns),
            "corr_map": dict(corr_map),
            "anchor": anchor,
            "tolerance": tolerance,
        },
        full_schema_reask=True,
    )

    validated = outcome.validated_output if isinstance(outcome.validated_output, dict) else None
    failed, kinds = _catch_telemetry(guard)

    user_facing = validated or {}
    return {
        "columns": [str(c) for c in (user_facing.get("columns_referenced") or [])],
        "claims": [_as_dict(c) for c in (user_facing.get("claims") or []) if _as_dict(c)],
        "verdict": str(user_facing.get("verdict", "")),
        # The Tier-B-catch analogue: how often a deterministic check fired.
        "rejections": failed,
        "rejection_kinds": kinds,
        # Loop cost, comparable with the contract's attempts_made.
        "provider_calls": calls["n"],
        "usage_in": calls["usage_in"] if calls["usage_seen"] else None,
        "usage_out": calls["usage_out"] if calls["usage_seen"] else None,
        "baseline": {
            "toolkit": GUARDRAILS_PACKAGE,
            "version": guardrails_version(),
            "arm": arm,
            "validators": _VALIDATOR_NAMES[arm],
            "num_reasks": num_reasks,
            "llm_calls": calls["n"],
            "validation_passed": bool(outcome.validation_passed),
            "outcome": "validated" if validated else "no_output",
            # What an application that ignored the failed validation would
            # have shipped. Recorded, never scored as the primary result.
            "passthrough": {
                "columns": [str(c) for c in (last_payload.get("columns_referenced") or [])],
                "claims": [_as_dict(c) for c in (last_payload.get("claims") or []) if _as_dict(c)],
                "verdict": str(last_payload.get("verdict", "")),
            },
        },
    }


def _catch_telemetry(guard: Any) -> tuple[int, list[str]]:
    """Total failed validations and their per-iteration channel composition.

    Mirrors the contract's ``schema_rejections`` / ``rejection_kinds`` so the
    two mechanisms are comparable on "how often did a check fire".
    """
    history = getattr(guard, "history", None)
    call = getattr(history, "last", None) if history is not None else None
    if call is None:
        return 0, []

    total = 0
    kinds: list[str] = []
    # ``ValidatorLogs.validator_name`` reports the CLASS name in guardrails
    # 0.11.0, not the registered rail alias, so both spellings are mapped.
    channel_of = {
        "EvidenceBoundColumns": "entity+omission",
        "mlcompass/evidence-bound-columns": "entity+omission",
        "EvidenceBoundClaims": "value",
        "mlcompass/evidence-bound-claims": "value",
        "AcceptableValues": "entity",
        "mlcompass/acceptable-values": "entity",
    }
    for iteration in call.iterations:
        failures = list(getattr(iteration, "failed_validations", []) or [])
        if not failures:
            continue
        total += len(failures)
        seen: list[str] = []
        for failure in failures:
            name = str(getattr(failure, "validator_name", "") or "")
            channel = channel_of.get(name, name or "structural")
            if channel not in seen:
                seen.append(channel)
        kinds.append("+".join(seen))
    return total, kinds
