"""The `guardrails_choices` arm, exercised without spending a provider call.

This arm exists to answer one objection: that 35.0 % measures a strawman,
because a competent Guardrails user would have taken the toolkit's own
acceptable-values check and populated it from the evidence. The arm is that
user, so two things have to be true of it and are asserted here.

It must actually fire on an out-of-evidence name -- an arm that silently passes
everything would produce 0/200 for the wrong reason and would make the
comparison meaningless. And it must be strictly weaker than Tier B: no value
tolerance, no anchor requirement. If it ever grew those, it would stop being
the generic check the objection is about and become a second Tier B.

The transport is a stub, so these run offline and cost nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

guardrails = pytest.importorskip("guardrails", reason="pip install '.[baselines]'")

import baseline_guardrails as gr  # noqa: E402

# `importorskip` above imports guardrails before any Guard is built, so the hub
# tracer initialises while still enabled and posts span batches to an endpoint
# that is unreachable offline, retrying with backoff. In a measurement run
# `build_guard` silences it first; here nothing has called it yet.
gr._silence_hub_telemetry()

ALLOWED = ["log_target_v2", "near_target_proxy", "feature_4", "feature_5"]
CORR = {"log_target_v2": 0.93, "near_target_proxy": 0.99, "feature_4": 0.01, "feature_5": 0.02}
ANCHOR = "log_target_v2"


def _transport_returning(payloads: list[dict[str, Any]]):
    """A stub provider that hands back the given payloads in order.

    The last one repeats, so an arm that keeps reasking past the supplied
    answers sees a stable response rather than an IndexError.
    """
    seen: list[list[dict[str, str]]] = []

    def transport(messages: list[dict[str, str]]):
        seen.append(messages)
        payload = payloads[min(len(seen) - 1, len(payloads) - 1)]
        return payload, 10, 10

    transport.seen = seen  # type: ignore[attr-defined]
    return transport


def _run(arm: str, payloads: list[dict[str, Any]], reasks: int = 2) -> dict[str, Any]:
    return gr.run_guardrails_once(
        _transport_returning(payloads),
        system_prompt="s",
        user_message="u",
        allowed_columns=ALLOWED,
        corr_map=CORR,
        anchor=ANCHOR,
        tolerance=0.005,
        arm=arm,
        num_reasks=reasks,
    )


def _answer(columns: list[str], claims: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "verdict": "leakage_confirmed",
        "confidence": "high",
        "columns_referenced": columns,
        "claims": claims if claims is not None else [],
        "narration": "n",
    }


def test_the_choices_arm_rejects_a_name_outside_the_evidence() -> None:
    """`r2` is the suspicious metric's name. It is in the evidence dictionary
    and it is not a column, which is the failure RQ1 reports at 42 %."""
    out = _run("guardrails_choices", [_answer([ANCHOR, "r2"])])
    assert out["rejections"] >= 1, "the acceptable-values check never fired"
    assert "entity" in " ".join(out["rejection_kinds"])
    assert "r2" not in out["columns"], "an out-of-evidence name reached the user"


def test_the_choices_arm_passes_a_clean_answer_on_the_first_attempt() -> None:
    """A check that rejects everything would also score 0/200, and for the
    wrong reason. One provider call, no rejections, the answer returned."""
    out = _run("guardrails_choices", [_answer([ANCHOR, "feature_4"])])
    assert out["rejections"] == 0
    assert out["provider_calls"] == 1
    assert set(out["columns"]) == {ANCHOR, "feature_4"}


def test_the_choices_arm_recovers_when_the_retry_fixes_the_answer() -> None:
    """The loop is the toolkit's; this asserts the arm uses it rather than
    failing closed on the first violation."""
    out = _run("guardrails_choices", [_answer([ANCHOR, "r2"]), _answer([ANCHOR])])
    assert out["provider_calls"] >= 2
    assert out["columns"] == [ANCHOR]
    assert out["baseline"]["validation_passed"]


def test_the_choices_arm_does_not_check_values() -> None:
    """Strictly weaker than Tier B, part one. A quoted number that contradicts
    the evidence by two orders of magnitude passes, because a generic
    acceptable-values check on one field cannot see the claims."""
    wrong = [{"column": ANCHOR, "statistic": "correlation", "value": 0.01}]
    out = _run("guardrails_choices", [_answer([ANCHOR], wrong)])
    assert out["rejections"] == 0
    assert out["claims"] and abs(out["claims"][0]["value"] - 0.01) < 1e-9


def test_the_choices_arm_does_not_require_the_anchor() -> None:
    """Strictly weaker than Tier B, part two. An answer that never addresses
    the top-ranked candidate passes."""
    out = _run("guardrails_choices", [_answer(["feature_5"])])
    assert out["rejections"] == 0
    assert out["columns"] == ["feature_5"]


def test_tier_b_catches_both_of_the_cases_choices_lets_through() -> None:
    """The pair is the point: same toolkit, same loop, same budget, and the
    difference between the arms is the difference between a generic check and
    a contract."""
    wrong = [{"column": ANCHOR, "statistic": "correlation", "value": 0.01}]
    value_case = _run("guardrails_tierb", [_answer([ANCHOR], wrong)])
    assert value_case["rejections"] >= 1
    assert "value" in " ".join(value_case["rejection_kinds"])

    anchor_case = _run("guardrails_tierb", [_answer(["feature_5"])])
    assert anchor_case["rejections"] >= 1


def test_every_arm_records_which_validators_it_attached() -> None:
    """A reader must never have to infer the configuration from the arm name."""
    for arm, expected in gr._VALIDATOR_NAMES.items():
        out = _run(arm, [_answer([ANCHOR])])
        assert out["baseline"]["arm"] == arm
        assert out["baseline"]["validators"] == expected


def test_the_stub_transport_is_what_the_guard_actually_called() -> None:
    """Guards the test itself: if the stub were bypassed, every assertion
    above would be vacuous."""
    transport = _transport_returning([_answer([ANCHOR, "r2"]), _answer([ANCHOR])])
    gr.run_guardrails_once(
        transport,
        system_prompt="s",
        user_message="u",
        allowed_columns=ALLOWED,
        corr_map=CORR,
        anchor=ANCHOR,
        tolerance=0.005,
        arm="guardrails_choices",
        num_reasks=2,
    )
    assert len(transport.seen) >= 2  # type: ignore[attr-defined]
    first = json.dumps(transport.seen[0])  # type: ignore[attr-defined]
    assert "u" in first
