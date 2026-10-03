"""The author-time lists of analysis plan A14 are nested and reproducible.

A14 runs the stale-enum arm with lists that cover 0, 3, 5, 8 and 10 of the
crowded instance's ten evidence columns. The comparison is only interpretable
if each list keeps its length, covers exactly k evidence columns, includes the
anchor whenever it covers any column, and covers every column that a smaller
list covers.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import reproduce_hallucination_ablation as h  # noqa: E402


def _crowded():
    ev = h.build_synthetic_crowded_evidence(seed=0)
    return list(h.evidence_allowed_columns(ev)), h.top_candidate(ev)


def test_lists_cover_k_columns_and_are_nested():
    allowed, anchor = _crowded()
    lists = h.coverage_lists(allowed, anchor)
    assert sorted(lists) == [0, 3, 5, 8, 10]
    covered = {k: set(v) & set(allowed) for k, v in lists.items()}
    for k, v in lists.items():
        assert len(v) == len(allowed) == 10
        assert len(covered[k]) == k
        assert (anchor in v) == (k > 0)
    ks = sorted(lists)
    for small, large in zip(ks, ks[1:], strict=False):
        assert covered[small] <= covered[large]
    assert lists[0] == tuple(sorted(h.STATIC_SCHEMA_COLUMNS))
    assert lists[10] == tuple(sorted(allowed))


def test_lists_and_orders_are_reproducible():
    allowed, anchor = _crowded()
    assert h.coverage_lists(allowed, anchor) == h.coverage_lists(allowed, anchor)
    labels = ["k0", "k3", "k5", "k8", "k10"]
    a, b = (
        h.interleave_coverage_orders(0, 200, labels),
        h.interleave_coverage_orders(0, 200, labels),
    )
    assert a == b
    assert all(sorted(o) == sorted(labels) for o in a)
    first = {label: sum(o[0] == label for o in a) for label in labels}
    assert all(20 < count < 70 for count in first.values())
