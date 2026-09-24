"""A short run of the registered falsifier (plan 2026-09 §1.6) on every test run.

The full run is 1,000,000 cases (scripts/property_falsifier.py, about 75 s);
this keeps 20,000 in the suite so a regression in verify() or strip_unsound()
fails CI rather than waiting for someone to run the script.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import property_falsifier as pf  # noqa: E402

from mlcompass.agents import evidence_contract as m  # noqa: E402


def test_no_generated_pair_breaks_c1_or_c2() -> None:
    rng = random.Random(20260924)
    for _ in range(20_000):
        bound = pf.gen_bound(m, rng)
        payload = pf.gen_payload(bound, rng)
        m.verify(payload, bound, m.PROFILE)
        cited, claims = m.strip_unsound(payload, bound, m.PROFILE)
        assert pf.oracle(bound, cited, claims) == []


def test_an_int_too_large_for_a_float_is_a_violation_not_a_crash() -> None:
    bound = m.BoundEvidence(
        domains={"columns_referenced": ("a",), "claims[].column": ("a",), "claims[].statistic": ("s",)},
        values={("a", "s"): 1.0},
        anchor="a",
    )
    payload = {"verdict": "leakage_likely", "columns_referenced": ["a"],
               "claims": [{"column": "a", "statistic": "s", "value": 10 ** 400}]}
    assert m.verify(payload, bound, m.PROFILE).value
    assert m.strip_unsound(payload, bound, m.PROFILE)[1] == []
