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
