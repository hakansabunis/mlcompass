"""The registered test A14 reproduces from the committed records.

Analysis plan A14 runs the stale-enum arm with author-time lists covering 0, 3,
5, 8 and 10 of the crowded instance's ten evidence columns, interleaved, 200
responses each. This test pins the reported counts and both registered tests
to the records in scripts/runs/2026-10-03_a14.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import analyze_a14 as a14  # noqa: E402


def test_a14_reproduces_the_reported_result():
    res = a14.analyse()
    cells = res["cells"]
    assert sorted(cells) == [0, 3, 5, 8, 10]
    assert all(c["n"] == 200 and c["transport_errors"] == 0 for c in cells.values())
    assert [cells[k]["invented_name"] for k in (0, 3, 5, 8, 10)] == [143, 2, 0, 3, 0]
    assert [cells[k]["coverage"] for k in (0, 3, 5, 8, 10)] == [0, 3, 5, 8, 10]
    assert res["P1_holds"] and res["P1_trend_p"] < 1e-70
    assert not res["P2_holds"] and abs(res["P2_fisher_p"] - 0.248) < 0.001
