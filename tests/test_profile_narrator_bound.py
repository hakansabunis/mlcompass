"""The profile narrator carries the same failure semantics as the leakage one.

A response with no admissible verdict is retried and, if it persists, aborted;
the host never substitutes an abstention for it (Proposition 1). The floor arm,
which verifies nothing, reports what the model wrote. Found missing in the
re-review of the TSE manuscript (NEW-11): the profile path used to clamp an
out-of-enum verdict to ``cannot_determine``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mlcompass.agents.evidence_contract import PROFILE  # noqa: E402
from mlcompass.agents.profile_narrator import (  # noqa: E402
    SUBMIT_TOOL_NAME,
    compact,
    narrate_profile_bound,
)


class _Fn:
    def __init__(self, args: dict) -> None:
        self.name = SUBMIT_TOOL_NAME
        self.arguments = json.dumps(args)


class _Call:
    def __init__(self, args: dict) -> None:
        self.function = _Fn(args)


class _Msg:
    def __init__(self, args: dict | None) -> None:
        self.tool_calls = [] if args is None else [_Call(args)]


class _Choice:
    def __init__(self, args: dict | None) -> None:
        self.message = _Msg(args)


class _Resp:
    def __init__(self, args: dict | None) -> None:
        self.choices = [_Choice(args)]


class FakeClient:
    def __init__(self, payloads: list) -> None:
        self._payloads = list(payloads)
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **_: object) -> _Resp:
        self.calls += 1
        return _Resp(self._payloads.pop(0))


@pytest.fixture(scope="module")
def evidence() -> dict:
    from reproduce_profile_battery import build_profile_evidence

    return build_profile_evidence(seed=0)


def _valid(evidence: dict) -> dict:
    bound = PROFILE.bind(compact(evidence))
    verdict = sorted(v for v in PROFILE.verdict_values if v != PROFILE.abstention)[0]
    return {
        "verdict": verdict,
        "confidence": "high",
        "columns_referenced": [bound.anchor],
        "claims": [],
        "narration": "ok",
    }


def test_invalid_verdict_then_valid_is_retried(evidence: dict) -> None:
    bad = {**_valid(evidence), "verdict": "definitely_messy"}
    client = FakeClient([bad, _valid(evidence)])
    out = narrate_profile_bound(evidence, client=client, max_retries=2)
    assert client.calls == 2
    assert out["rejection_kinds"] == ["malformed"]
    assert out["aborted"] is False
    assert out["verdict"] in PROFILE.verdict_values


def test_persistent_missing_tool_call_aborts(evidence: dict) -> None:
    client = FakeClient([None, None, None])
    out = narrate_profile_bound(evidence, client=client, max_retries=2)
    assert client.calls == 3
    assert out["aborted"] is True
    assert out["verdict"] == ""
    assert out["claims"] == [] and out["columns_referenced"] == []
    assert out["omitted_critical_evidence"] is False


def test_floor_arm_reports_the_model_verbatim(evidence: dict) -> None:
    bad = {**_valid(evidence), "verdict": "definitely_messy"}
    client = FakeClient([bad])
    out = narrate_profile_bound(evidence, client=client, verify_response=False)
    assert client.calls == 1
    assert out["aborted"] is False
    assert out["verdict"] == "definitely_messy"
