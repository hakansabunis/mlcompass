"""Fault injection through the shipped delivery path, down to what the user sees.

No live response in the study ever reached Tier B's last resort: stripping, the
omission flag and the abort. The EMSE review of 2026-10-01 asked for those
branches to be driven through the real path, not only through the verifier
functions. Each test here scripts a narrator that keeps the same fault on every
attempt, runs the shipped ``investigate_leakage_bound`` until the retry budget
is spent, renders the result with the shipped panel, and checks three things
together: the delivered payload (schema-shaped, sound on C1-C2), the flags, and
the text the engineer would read.
"""

from __future__ import annotations

import json
import math

import pytest
from agentlite.testing import MockClient, tool_use_response
from rich.console import Console

from mlcompass.agents.evidence_contract import LEAKAGE, bind_leakage, verify
from mlcompass.agents.leakage_investigator import SUBMIT_TOOL_NAME, investigate_leakage_bound
from mlcompass.ui.evaluate import render_leakage_narration

EVIDENCE: dict = {
    "row_count": 1000,
    "trustworthy_sample_size": True,
    "suspicious_metric": {"name": "r2", "value": 1.0},
    "target_feature_correlations": [
        {"feature": "log_target_v2", "correlation": 0.999, "method": "spearman"},
        {"feature": "near_target_proxy", "correlation": 0.95, "method": "pearson"},
        {"feature": "feature_3", "correlation": 0.21, "method": "pearson"},
    ],
    "perfect_match_rate": 0.0,
    "candidate_leak_columns": ["log_target_v2"],
    "notes": [],
}
GOOD_CLAIM = {"column": "log_target_v2", "statistic": "correlation", "value": 0.999}


def _payload(**over) -> dict:
    p = {
        "verdict": "leakage_likely",
        "confidence": "high",
        "columns_referenced": ["log_target_v2"],
        "claims": [dict(GOOD_CLAIM)],
        "narration": "The leak column tracks the target.",
        "recommended_checks": ["drop log_target_v2 and retrain"],
    }
    p.update(over)
    return p


def _run(payload: dict, attempts: int = 3) -> tuple[dict, str]:
    """Return the delivered narration and the panel text after the budget."""
    client = MockClient(
        responses=[
            tool_use_response(SUBMIT_TOOL_NAME, json.loads(json.dumps(payload, default=str)))
            if not _has_nonfinite(payload)
            else tool_use_response(SUBMIT_TOOL_NAME, payload)
            for _ in range(attempts)
        ]
    )
    out = investigate_leakage_bound(EVIDENCE, client=client, max_retries=attempts - 1)
    console = Console(record=True, width=200)
    render_leakage_narration(console, out)
    return out, console.export_text()


def _has_nonfinite(p: dict) -> bool:
    for c in p.get("claims") or []:
        v = c.get("value") if isinstance(c, dict) else None
        if isinstance(v, float) and not math.isfinite(v):
            return True
        if isinstance(v, int) and abs(v) > 10**300:
            return True
    return False


def _delivered_is_sound(out: dict) -> None:
    """C1 and C2 hold on what reaches the user, and every claim is whole."""
    bound = bind_leakage(EVIDENCE)
    v = verify(
        {
            "verdict": out["verdict"],
            "columns_referenced": out["columns_referenced"],
            "claims": out["claims"],
        },
        bound,
        LEAKAGE,
    )
    assert not v.entity and not v.value
    for c in out["claims"]:
        assert set(c) >= {"column", "statistic", "value"}, "deletion removes whole claims"


@pytest.mark.parametrize(
    ("label", "payload", "gone"),
    [
        ("C1 cited column", _payload(columns_referenced=["log_target_v2", "r2"]), "r2"),
        (
            "C1 claim column",
            _payload(
                claims=[
                    dict(GOOD_CLAIM),
                    {"column": "ghost", "statistic": "correlation", "value": 0.5},
                ]
            ),
            "ghost",
        ),
        (
            "C2 wrong number",
            _payload(
                claims=[
                    dict(GOOD_CLAIM),
                    {"column": "feature_3", "statistic": "correlation", "value": 0.77},
                ]
            ),
            "0.77",
        ),
        (
            "C2 non-finite",
            _payload(
                claims=[
                    dict(GOOD_CLAIM),
                    {"column": "feature_3", "statistic": "correlation", "value": float("nan")},
                ]
            ),
            "nan",
        ),
        (
            "C2 400-digit integer",
            _payload(
                claims=[
                    dict(GOOD_CLAIM),
                    {"column": "feature_3", "statistic": "correlation", "value": 10**400},
                ]
            ),
            "0000000000",
        ),
    ],
)
def test_a_persistent_soundness_fault_is_stripped_and_announced(label, payload, gone) -> None:
    out, panel = _run(payload)
    assert out["aborted"] is False, label
    assert out["had_unrecoverable_violation"] is True, label
    _delivered_is_sound(out)
    assert "Content stripped" in panel, label
    verified = panel.split("Hypothesis")[0]  # the verified part of the panel
    assert gone not in verified, f"{label}: the stripped item reached the verified panel"
    assert "log_target_v2" in verified, f"{label}: the sound content was lost"


def test_a_persistent_omission_is_delivered_flagged_with_the_anchor_named() -> None:
    out, panel = _run(
        _payload(
            columns_referenced=["feature_3"],
            claims=[{"column": "feature_3", "statistic": "correlation", "value": 0.21}],
        )
    )
    assert out["aborted"] is False
    assert out["omitted_critical_evidence"] is True
    _delivered_is_sound(out)
    assert "Incomplete" in panel and "log_target_v2" in panel


def test_soundness_and_omission_together() -> None:
    out, panel = _run(
        _payload(
            columns_referenced=["feature_3", "r2"],
            claims=[{"column": "feature_3", "statistic": "correlation", "value": 0.21}],
        )
    )
    assert out["had_unrecoverable_violation"] is True and out["omitted_critical_evidence"] is True
    _delivered_is_sound(out)
    assert "Content stripped" in panel and "Incomplete" in panel


def test_a_persistent_inadmissible_verdict_aborts_and_shows_nothing() -> None:
    out, panel = _run(_payload(verdict="definitely_leaking"))
    assert out["aborted"] is True
    assert out["columns_referenced"] == [] and out["claims"] == []
    assert "No narration" in panel
    assert "log_target_v2" not in panel and "The leak column" not in panel


def test_a_missing_tool_call_aborts() -> None:
    client = MockClient(responses=[tool_use_response("some_other_tool", {}) for _ in range(3)])
    out = investigate_leakage_bound(EVIDENCE, client=client, max_retries=2)
    console = Console(record=True, width=200)
    render_leakage_narration(console, out)
    assert out["aborted"] is True
    assert "No narration" in console.export_text()


def test_a_fault_repaired_on_retry_is_delivered_clean_and_unflagged() -> None:
    client = MockClient(
        responses=[
            tool_use_response(
                SUBMIT_TOOL_NAME, _payload(columns_referenced=["log_target_v2", "r2"])
            ),
            tool_use_response(SUBMIT_TOOL_NAME, _payload()),
        ]
    )
    out = investigate_leakage_bound(EVIDENCE, client=client, max_retries=2)
    console = Console(record=True, width=200)
    render_leakage_narration(console, out)
    panel = console.export_text()
    assert out["had_unrecoverable_violation"] is False and out["schema_rejections"] == 1
    _delivered_is_sound(out)
    assert "Content stripped" not in panel and "Incomplete" not in panel
