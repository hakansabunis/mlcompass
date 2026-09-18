"""Field-indexed admissible domains, and a verifier that does not know the task.

The paper this artifact accompanies defines :math:`A_{E,f}` --- the admissible
set for narration field *f*, derived from the evidence *E* at call time --- and
claims that three invariants make the contract portable: a deterministic
evidence producer, a finite enumerable set of observed entities and values, and
a runtime enum bound to that set with independent verification.

The shipped leakage path implements a *correct instance* of that, not the class.
It carries one column set and freezes ``statistic`` to ``("correlation",)``
because leakage evidence happens to carry exactly one kind of statistic. Its
value check reads ``values[column]`` and never consults the statistic field at
all, which is sound only while there is one statistic to confuse. Bind a second
evidence shape to it and both facts become defects: a profile carries a dozen
statistics per column, and comparing a claimed ``mean`` against a stored
``missing_pct`` is exactly the scorer defect the paper reports in its own
instruments section --- this time in production code.

So this module is the abstraction the paper asserts, extracted so it can be
instantiated more than once. A :class:`ContractSpec` says how to read one
evidence shape. :meth:`ContractSpec.bind` turns evidence into field-indexed
domains, a ``(entity, statistic) -> value`` table and a completeness anchor.
:func:`verify` checks a narration against those three and knows nothing about
leakage, profiles, or whatever comes next.

Nothing here changes what the leakage contract does. ``contracts.LEAKAGE``
reproduces the shipped behaviour, and the existing test suite is the proof.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BoundEvidence",
    "ContractSpec",
    "Violations",
    "build_claim_schema",
    "build_payload_schema",
    "correction_text",
    "strip_unsound",
    "verify",
]


# --------------------------------------------------------------------------- #
# The bound evidence: what a narration is allowed to say, computed per call.   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BoundEvidence:
    """One evidence dictionary, resolved into everything the contract checks.

    Attributes:
        domains: ``field -> A_{E,f}``. The field names are narration field
            names (``"columns_referenced"``, ``"claims[].statistic"``), and
            the tuples are sorted so schema generation is deterministic and a
            run is reproducible.
        values: ``(entity, statistic) -> measured value``. Keyed on the pair
            and not on the entity, because an entity generally carries more
            than one measured quantity and checking the wrong one silently
            passes a fabricated number.
        anchor: The completeness anchor --- the one evidence item a committed
            verdict must address --- or None when the evidence names none.
        anchor_description: How to name the anchor in a corrective retry. The
            narrator is told which item it skipped and why that item matters,
            because a rejection that cannot say what was missing is the
            content-free control, not the treatment.
        tolerance: Absolute tolerance for the value channel.
    """

    domains: Mapping[str, tuple[str, ...]]
    values: Mapping[tuple[str, str], float]
    anchor: str | None
    anchor_description: str = "the top-ranked evidence item"
    tolerance: float = 0.005

    def admissible(self, field_name: str) -> frozenset[str]:
        """The admissible set for one field, empty when the field is unbound."""
        return frozenset(self.domains.get(field_name, ()))


# --------------------------------------------------------------------------- #
# The spec: how to read one evidence shape, and which fields Tier A gates.     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ContractSpec:
    """How to bind one evidence shape, and what its narration fields are called.

    Attributes:
        name: Task identifier, used in telemetry and run records.
        binder: ``evidence -> BoundEvidence``. The only task-specific code.
        cited_field: The narration's array-of-entity field.
        claim_entity_field: Key inside a claim object holding the entity.
        claim_statistic_field: Key inside a claim object naming the quantity.
        claim_value_field: Key inside a claim object holding the number.
        entity_noun: Singular noun for the entity kind, for correction text.
        tier_a_fields: Which fields lose their enum when Tier A is disabled.
            The stress configuration drops these to make Tier B's catches
            observable in isolation; any field *not* listed keeps its enum in
            every configuration. Leakage lists its two column fields and not
            ``claims[].statistic``, which is how the shipped arm behaves and
            therefore what the published measurements measured.
        verdict_values: Admissible verdicts.
        confidence_values: Admissible confidence levels.
        abstention: The verdict that means "no commitment", exempt from the
            completeness check.
    """

    name: str
    binder: Callable[[Mapping[str, Any]], BoundEvidence]
    cited_field: str = "columns_referenced"
    claim_entity_field: str = "column"
    claim_statistic_field: str = "statistic"
    claim_value_field: str = "value"
    entity_noun: str = "column"
    tier_a_fields: frozenset[str] = frozenset({"columns_referenced", "claims[].column"})
    verdict_values: frozenset[str] = frozenset()
    confidence_values: frozenset[str] = frozenset()
    abstention: str = "cannot_determine"

    @property
    def claim_entity_path(self) -> str:
        return f"claims[].{self.claim_entity_field}"

    @property
    def claim_statistic_path(self) -> str:
        return f"claims[].{self.claim_statistic_field}"

    def bind(self, evidence: Mapping[str, Any]) -> BoundEvidence:
        """Resolve ``evidence`` into its admissible domains, values and anchor."""
        return self.binder(evidence)


# --------------------------------------------------------------------------- #
# Tier B: three channels, none of which know what the evidence is about.      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Violations:
    """What the three channels found, one round."""

    entity: tuple[str, ...] = ()
    value: tuple[str, ...] = ()
    omitted: bool = False

    def __bool__(self) -> bool:
        return bool(self.entity) or bool(self.value) or self.omitted

    @property
    def kinds(self) -> str:
        """The composition telemetry string, e.g. ``"entity+omission"``."""
        parts = []
        if self.entity:
            parts.append("entity")
        if self.value:
            parts.append("value")
        if self.omitted:
            parts.append("omission")
        return "+".join(parts)


def _claims_of(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [c for c in (payload.get("claims") or []) if isinstance(c, dict)]


def _cited_of(payload: Mapping[str, Any], spec: ContractSpec) -> list[str]:
    return [str(c) for c in (payload.get(spec.cited_field) or [])]


def _referenced(payload: Mapping[str, Any], spec: ContractSpec) -> set[str]:
    """Every entity the narration points at, cited or claimed."""
    out = set(_cited_of(payload, spec))
    for claim in _claims_of(payload):
        out.add(str(claim.get(spec.claim_entity_field, "")))
    return out


def is_committed(payload: Mapping[str, Any], spec: ContractSpec) -> bool:
    """True when the narration takes a position the completeness rule applies to."""
    verdict = str(payload.get("verdict", spec.abstention))
    return bool(payload) and verdict in spec.verdict_values and verdict != spec.abstention


def verify(
    payload: Mapping[str, Any], bound: BoundEvidence, spec: ContractSpec
) -> Violations:
    """Check one narration payload on all three channels.

    Entity soundness: every cited entity is in :math:`A_{E,f}` for the field
    that carries it. Value soundness: every structured claim names a measured
    ``(entity, statistic)`` pair and reports it within tolerance. Completeness:
    a committed verdict addresses the anchor.
    """
    cited_domain = bound.admissible(spec.cited_field)
    entity = tuple(c for c in _cited_of(payload, spec) if c not in cited_domain)

    claim_entity_domain = bound.admissible(spec.claim_entity_path)
    stat_domain = bound.admissible(spec.claim_statistic_path)

    value: list[str] = []
    for claim in _claims_of(payload):
        ent = str(claim.get(spec.claim_entity_field, ""))
        stat = str(claim.get(spec.claim_statistic_field, ""))
        val = claim.get(spec.claim_value_field)
        if claim_entity_domain and ent not in claim_entity_domain:
            value.append(f"{ent}: not in evidence")
            continue
        if stat_domain and stat not in stat_domain:
            value.append(f"{ent}.{stat}: not a quantity the evidence carries")
            continue
        measured = bound.values.get((ent, stat))
        if measured is None:
            value.append(f"{ent}.{stat}: not in evidence")
        elif (
            not isinstance(val, (int, float))
            or isinstance(val, bool)
            or abs(float(val) - measured) > bound.tolerance
        ):
            value.append(f"{ent}.{stat}: cited {val}, evidence says {measured:.4f}")

    omitted = (
        is_committed(payload, spec)
        and bound.anchor is not None
        and bound.anchor not in _referenced(payload, spec)
    )
    return Violations(entity=entity, value=tuple(value), omitted=omitted)


def strip_unsound(
    payload: Mapping[str, Any], bound: BoundEvidence, spec: ContractSpec
) -> tuple[list[str], list[dict[str, Any]]]:
    """Delete everything still unsound after the retry budget.

    Deletion is the only safe repair (the paper's asymmetry proposition):
    removing an unsupported citation cannot fabricate attribution, whereas
    inserting a missing one would put words in the narrator's mouth. So
    omissions are never repaired here --- they are flagged by the caller.
    """
    cited_domain = bound.admissible(spec.cited_field)
    claim_domain = bound.admissible(spec.claim_entity_path)
    stat_domain = bound.admissible(spec.claim_statistic_path)

    cited_clean = [c for c in _cited_of(payload, spec) if c in cited_domain]

    claims_clean: list[dict[str, Any]] = []
    for claim in _claims_of(payload):
        ent = str(claim.get(spec.claim_entity_field, ""))
        stat = str(claim.get(spec.claim_statistic_field, ""))
        val = claim.get(spec.claim_value_field)
        if claim_domain and ent not in claim_domain:
            continue
        if stat_domain and stat not in stat_domain:
            continue
        measured = bound.values.get((ent, stat))
        if measured is None:
            continue
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        if abs(float(val) - measured) > bound.tolerance:
            continue
        claims_clean.append(claim)

    return cited_clean, claims_clean


def correction_text(
    violations: Violations, bound: BoundEvidence, spec: ContractSpec, *, tool_name: str
) -> str:
    """The corrective retry, with the offending items named.

    Naming them is the treatment. The content-free variant is a separate
    control and is built by the caller, not here.
    """
    parts: list[str] = []
    if violations.entity:
        allowed = list(bound.domains.get(spec.cited_field, ()))
        parts.append(
            f"you cited {spec.entity_noun}s NOT in the evidence dictionary: "
            f"{list(violations.entity)}; you may cite only these "
            f"{spec.entity_noun}s: {allowed}"
        )
    if violations.value:
        parts.append(
            "these claims do not match the measured values: "
            + "; ".join(violations.value)
            + " — copy values exactly from the evidence"
        )
    if violations.omitted:
        parts.append(
            f"you committed to a verdict but did not address {bound.anchor_description} "
            f"'{bound.anchor}' — address it or answer {spec.abstention}"
        )
    return (
        "\n\nYour previous answer violated the contract: "
        + ". Also, ".join(parts)
        + f". Re-answer through {tool_name}."
    )


# --------------------------------------------------------------------------- #
# Tier A: the schema, with every evidence-derived enum bound at call time.     #
# --------------------------------------------------------------------------- #


def _string_schema(
    values: Sequence[str] | None, *, enumerate_it: bool
) -> dict[str, Any]:
    if enumerate_it and values is not None:
        return {"type": "string", "enum": list(values)}
    return {"type": "string"}


def build_claim_schema(
    bound: BoundEvidence, spec: ContractSpec, *, enforce_enum: bool
) -> dict[str, Any]:
    """The ``{entity, statistic, value}`` claim object, enums bound from E."""
    ent_enum = enforce_enum or spec.claim_entity_path not in spec.tier_a_fields
    stat_enum = enforce_enum or spec.claim_statistic_path not in spec.tier_a_fields
    return {
        "type": "object",
        "properties": {
            spec.claim_entity_field: _string_schema(
                bound.domains.get(spec.claim_entity_path), enumerate_it=ent_enum
            ),
            spec.claim_statistic_field: _string_schema(
                bound.domains.get(spec.claim_statistic_path), enumerate_it=stat_enum
            ),
            spec.claim_value_field: {"type": "number"},
        },
        "required": [
            spec.claim_entity_field,
            spec.claim_statistic_field,
            spec.claim_value_field,
        ],
    }


def build_payload_schema(
    bound: BoundEvidence,
    spec: ContractSpec,
    *,
    enforce_enum: bool = True,
    strict: bool = False,
    extra_properties: Mapping[str, Any] | None = None,
    extra_required: Sequence[str] = (),
) -> dict[str, Any]:
    """The narration schema, provider-neutral.

    ``enforce_enum=False`` drops the enums on :attr:`ContractSpec.tier_a_fields`
    and keeps every other field's, which is what the stress configuration
    measures: the field structure without the call-time binding, so Tier B's
    catches become observable on their own.
    """
    cited_enum = enforce_enum or spec.cited_field not in spec.tier_a_fields
    claim_schema = build_claim_schema(bound, spec, enforce_enum=enforce_enum)

    properties: dict[str, Any] = {
        "verdict": {"type": "string", "enum": sorted(spec.verdict_values)},
        "confidence": {"type": "string", "enum": sorted(spec.confidence_values)},
        spec.cited_field: {
            "type": "array",
            "items": _string_schema(
                bound.domains.get(spec.cited_field), enumerate_it=cited_enum
            ),
        },
        "claims": {"type": "array", "items": claim_schema},
        "narration": {"type": "string"},
        "recommended_checks": {"type": "array", "items": {"type": "string"}},
    }
    if extra_properties:
        properties.update(dict(extra_properties))

    required = ["verdict", "confidence", spec.cited_field, "claims", "narration"]
    required.extend(extra_required)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    if strict:
        schema["required"] = sorted(properties)
        schema["additionalProperties"] = False
        claim_schema["additionalProperties"] = False
    return schema


# --------------------------------------------------------------------------- #
# Binders. One per evidence shape; everything above is shared.                 #
# --------------------------------------------------------------------------- #


def _sorted_unique(values: Any) -> tuple[str, ...]:
    out: set[str] = set()
    for v in values or ():
        if v is not None:
            out.add(str(v))
    return tuple(sorted(out))


def bind_leakage(evidence: Mapping[str, Any]) -> BoundEvidence:
    """Bind ``mlcompass.tools.leakage.detect_leakage`` output.

    Reproduces the shipped behaviour exactly: the column domain is the union of
    the candidate leak columns and every feature the correlation pass reported,
    the value table is that pass keyed on ``(feature, "correlation")``, and the
    anchor is the top-ranked candidate.
    """
    columns: set[str] = set()
    for col in evidence.get("candidate_leak_columns") or []:
        if col is not None:
            columns.add(str(col))

    values: dict[tuple[str, str], float] = {}
    for entry in evidence.get("target_feature_correlations") or []:
        if not isinstance(entry, dict):
            continue
        feature = entry.get("feature")
        if feature is None:
            continue
        name = str(feature)
        columns.add(name)
        corr = entry.get("correlation")
        if isinstance(corr, (int, float)) and not isinstance(corr, bool):
            values[(name, "correlation")] = float(corr)

    domain = tuple(sorted(columns))
    candidates = evidence.get("candidate_leak_columns") or []
    return BoundEvidence(
        domains={
            "columns_referenced": domain,
            "claims[].column": domain,
            "claims[].statistic": ("correlation",),
        },
        values=values,
        anchor=str(candidates[0]) if candidates else None,
        anchor_description="the top-ranked candidate-leak column",
    )


#: Every numeric quantity ``analyze_dataset`` reports per column. The profile
#: contract's statistic domain is the subset actually present, computed per
#: call --- a categorical column carries ``cardinality`` and no ``mean``, so a
#: single frozen list would admit statistics the evidence never measured.
PROFILE_STATISTICS: tuple[str, ...] = (
    "cardinality",
    "iqr_outliers",
    "max",
    "mean",
    "min",
    "missing_count",
    "missing_pct",
    "q25",
    "q50",
    "q75",
    "std",
    "z_score_outliers",
    "zero_ratio",
)


def bind_profile(evidence: Mapping[str, Any]) -> BoundEvidence:
    """Bind ``mlcompass.tools.dataset.analyze_dataset`` output.

    A second evidence shape, and a genuinely different one. The column domain
    is *every* column rather than a filtered candidate list, so the enum is as
    wide as the frame. The statistic domain is a dozen names rather than one,
    and it is computed from what each column actually carries. And the value
    table must be keyed on the pair: ``mean`` and ``missing_pct`` are both
    floats on the same column, so an entity-keyed table would compare a claimed
    mean against a stored missing rate and pass it.

    The anchor is the column with the most missing data --- the item a profile
    narration that commits to a data-quality verdict cannot responsibly skip.
    """
    columns: list[str] = []
    values: dict[tuple[str, str], float] = {}
    statistics: set[str] = set()

    worst_missing: tuple[float, str] | None = None

    for entry in evidence.get("columns") or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if name is None:
            continue
        col = str(name)
        columns.append(col)

        def put(stat: str, raw: Any) -> None:
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                values[(col, stat)] = float(raw)
                statistics.add(stat)

        put("missing_count", entry.get("missing_count"))
        put("missing_pct", entry.get("missing_pct"))
        put("zero_ratio", entry.get("zero_ratio"))
        put("cardinality", entry.get("cardinality"))

        stats = entry.get("stats")
        if isinstance(stats, dict):
            for stat in ("mean", "std", "min", "max", "q25", "q50", "q75"):
                put(stat, stats.get(stat))

        outliers = entry.get("outliers")
        if isinstance(outliers, dict):
            put("iqr_outliers", outliers.get("iqr_count"))
            put("z_score_outliers", outliers.get("z_score_count"))

        pct = entry.get("missing_pct")
        if isinstance(pct, (int, float)) and not isinstance(pct, bool):
            if worst_missing is None or float(pct) > worst_missing[0]:
                worst_missing = (float(pct), col)

    domain = tuple(sorted(set(columns)))
    return BoundEvidence(
        domains={
            "columns_referenced": domain,
            "claims[].column": domain,
            "claims[].statistic": tuple(sorted(statistics)),
        },
        values=values,
        anchor=worst_missing[1] if worst_missing else None,
        anchor_description="the column carrying the most missing data",
    )


LEAKAGE_VERDICTS = frozenset({"leakage_likely", "leakage_unlikely", "cannot_determine"})
PROFILE_VERDICTS = frozenset({"ready_to_train", "needs_cleaning", "cannot_determine"})
CONFIDENCE_VALUES = frozenset({"high", "medium", "low", "cannot_determine"})

#: The shipped leakage contract, re-expressed. Behaviour is unchanged; the
#: existing test suite is the check on that claim.
LEAKAGE = ContractSpec(
    name="leakage",
    binder=bind_leakage,
    entity_noun="column",
    verdict_values=LEAKAGE_VERDICTS,
    confidence_values=CONFIDENCE_VALUES,
)

#: The second instance: dataset profiling. Same three channels, same verifier,
#: a wider entity domain and a statistic domain that is not a singleton.
PROFILE = ContractSpec(
    name="profile",
    binder=bind_profile,
    entity_noun="column",
    tier_a_fields=frozenset(
        {"columns_referenced", "claims[].column", "claims[].statistic"}
    ),
    verdict_values=PROFILE_VERDICTS,
    confidence_values=CONFIDENCE_VALUES,
)

CONTRACTS: dict[str, ContractSpec] = {LEAKAGE.name: LEAKAGE, PROFILE.name: PROFILE}
