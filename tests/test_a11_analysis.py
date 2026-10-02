"""The registered test A11 reproduces from the committed records.

Analysis plan A11 compares the strong profile arms with and without the
verifier, interleaved, N = 600 each, by a two-sided Fisher exact test with the
Newcombe interval of the difference. This test pins the reported result to the
records in scripts/runs/profile/2026-10-02_a11.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import analyze_a11 as a11  # noqa: E402


def test_a11_reproduces_the_reported_result():
    res = a11.analyse()
    assert res["complete_pairs"] == 600
    unverified, verified = res["cells"]["strong_noverify"], res["cells"]["strong_verify"]
    assert (unverified["n"], unverified["delivered_violations"]) == (600, 10)
    assert (verified["n"], verified["delivered_violations"]) == (600, 0)
    assert verified["rejected_attempts"] == verified["confirmed_by_second_reading"] == 11
    assert verified["provider_calls"] == 611
    assert res["rejects_at_0.05"]
    assert abs(res["fisher_two_sided_p"] - 0.00188) < 1e-4
    low, high = res["newcombe_95"]
    assert 0.0065 < low < 0.0070 and 0.0300 < high < 0.0308
