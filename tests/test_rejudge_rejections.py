"""Every rejection that the profile records keep is a real violation.

The paper's rejection audit (Table 11) rests on scripts/rejudge_rejections.py,
which reads only the evidence dumps and the stored attempts. This test pins its
result to the committed records: thirteen rejected attempts, each one a
violation by the independent reading, each repaired by the next attempt.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import rejudge_rejections as rj  # noqa: E402


def test_every_stored_rejection_is_confirmed_and_repaired():
    rows, total, agree = rj.rejudge()
    assert total == 13
    assert agree == 13
    assert all(r["finding"] != "nothing found" for r in rows)
    assert all(r["final_clean"] for r in rows)
    assert all(r["fate"] in {"dropped", "corrected", "verdict supplied"} for r in rows)


def test_wrong_numbers_are_the_suffix_siblings_values():
    rows, _, _ = rj.rejudge()
    wrong = [r for r in rows if r["finding"].startswith("wrong number")]
    assert len({(r["run"], r["i"]) for r in wrong}) == 7
    for r in wrong:
        col = r["item"].split(".", 1)[0]
        sibling = col.replace("num_feature", "cat_feature")
        assert f"the value of {sibling}." in r["finding"]


LEAK_EVIDENCE = {
    "suspicious_metric": {"name": "r2", "value": 1.0},
    "target_feature_correlations": [
        {"feature": "log_target_v2", "correlation": 0.999, "method": "spearman"},
        {"feature": "feature_3", "correlation": 0.21, "method": "pearson"},
    ],
    "candidate_leak_columns": ["log_target_v2"],
    "perfect_match_rate": 0.0,
}


def test_leakage_attempts_are_judged_by_the_second_scorer():
    rejected = {
        "columns_referenced": ["log_target_v2", "r2"],
        "verdict": "leakage_likely",
        "claims": [{"column": "log_target_v2", "statistic": "correlation", "value": 0.5}],
    }
    repaired = {
        "columns_referenced": ["log_target_v2"],
        "verdict": "leakage_likely",
        "claims": [{"column": "log_target_v2", "statistic": "correlation", "value": 0.999}],
    }
    found = rj.judge_leakage_attempt(rejected, LEAK_EVIDENCE)
    assert "name outside E: r2" in found and "wrong number" in found
    assert rj.judge_leakage_attempt(repaired, LEAK_EVIDENCE) == []
    assert rj.judge_leakage_attempt({"verdict": ""}, LEAK_EVIDENCE) == ["no admissible verdict"]


def test_interleaved_orders_are_balanced_and_reproducible():
    import reproduce_profile_battery as pb

    a, b = pb.interleave_orders(0, 200), pb.interleave_orders(0, 200)
    assert a == b
    assert all(sorted(o) == ["natural", "suffixed"] for o in a)
    first_suffixed = sum(o[0] == "suffixed" for o in a)
    assert 70 < first_suffixed < 130
