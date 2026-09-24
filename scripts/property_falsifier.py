"""The pre-registered falsifier for the mechanism claim (analysis plan 2026-09 §1.6).

Registered before the Round-2 runs and not built until the pre-submission review
asked where it was. It generates (bound evidence, candidate response) pairs and
asserts the invariant the manuscript states for Tier B: after strip_unsound,
no cited entity lies outside its field's admissible set (C1), and no claim lies
outside dom(V_E) or outside tolerance (C2). The oracle below is written
independently of the verifier: it does not call verify(), so a defect shared by
verify() and strip_unsound() cannot hide itself.

A single failing pair refutes the claim for the code under test (§1.6). Run
against an older revision with --module to reproduce the refutation the review
found by hand: before commit dab4130 a NaN claim value passed both functions.

    python -X utf8 scripts/property_falsifier.py [--cases 1000000] [--seed 0]
    python -X utf8 scripts/property_falsifier.py --module path/to/evidence_contract.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import random
import string
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_contract(path: pathlib.Path | None):
    if path is None:
        sys.path.insert(0, str(ROOT / "src"))
        from mlcompass.agents import evidence_contract as m  # noqa: PLC0415

        return m
    spec = importlib.util.spec_from_file_location("contract_under_test", path)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["contract_under_test"] = m
    spec.loader.exec_module(m)
    return m


def rand_name(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_lowercase + "_0123456789") for _ in range(rng.randint(1, 8)))


def gen_bound(m, rng: random.Random):
    cols = sorted({rand_name(rng) for _ in range(rng.randint(1, 12))})
    stats = sorted({rand_name(rng) for _ in range(rng.randint(1, 5))})
    values = {}
    for c in cols:
        for s in stats:
            if rng.random() < 0.6:
                values[(c, s)] = rng.choice([0.0, 1.0, -1.0, rng.uniform(-1, 1), rng.uniform(-1e4, 1e4)])
    return m.BoundEvidence(
        domains={
            "columns_referenced": tuple(cols),
            "claims[].column": tuple(cols),
            "claims[].statistic": tuple(stats),
        },
        values=values,
        anchor=rng.choice(cols),
    )


def gen_token(rng: random.Random, pool: list):
    r = rng.random()
    if r < 0.55 and pool:
        return rng.choice(pool)
    if r < 0.85:
        return rand_name(rng)
    return rng.choice([None, 3, 2.5, True, "", " ", "None"])


def gen_value(rng: random.Random, measured: float | None):
    base = measured if measured is not None else rng.uniform(-10, 10)
    return rng.choice([
        base,
        base + rng.uniform(-0.005, 0.005),
        base + rng.choice([-1, 1]) * rng.uniform(0.0050001, 0.01),
        base + rng.uniform(-100, 100),
        -base,
        float("nan"), float("inf"), float("-inf"),
        True, False, None, str(base), [base], 10 ** 400 if rng.random() < 0.01 else base,
    ])


def gen_payload(bound, rng: random.Random) -> dict:
    cols = list(bound.domains["claims[].column"])
    stats = list(bound.domains["claims[].statistic"])
    cited = [gen_token(rng, cols) for _ in range(rng.randint(0, 6))]
    claims = []
    for _ in range(rng.randint(0, 6)):
        c = gen_token(rng, cols)
        s = gen_token(rng, stats)
        measured = bound.values.get((str(c), str(s)))
        claim = {"column": c, "statistic": s, "value": gen_value(rng, measured)}
        if rng.random() < 0.05:
            claim.pop(rng.choice(["column", "statistic", "value"]))
        claims.append(claim)
    if rng.random() < 0.05:
        claims.append(rng.choice(["not a dict", 7, None]))
    return {"verdict": "leakage_likely", "columns_referenced": cited, "claims": claims}


def oracle(bound, cited_clean, claims_clean) -> list[str]:
    """Independent statement of (C1) and (C2) on the returned structured channels."""
    bad = []
    a_cited = set(bound.domains["columns_referenced"])
    a_col = set(bound.domains["claims[].column"])
    a_stat = set(bound.domains["claims[].statistic"])
    for x in cited_clean:
        if str(x) not in a_cited:
            bad.append(f"C1 cited {x!r}")
    for c in claims_clean:
        e, s, v = str(c.get("column", "")), str(c.get("statistic", "")), c.get("value")
        if e not in a_col or s not in a_stat or (e, s) not in bound.values:
            bad.append(f"C2 domain {c!r}")
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            bad.append(f"C2 type {c!r}")
            continue
        try:
            fv = float(v)
        except OverflowError:
            bad.append(f"C2 overflow {c!r}")
            continue
        if not math.isfinite(fv) or abs(fv - bound.values[(e, s)]) > bound.tolerance:
            bad.append(f"C2 value {c!r} vs {bound.values[(e, s)]!r}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--module", type=pathlib.Path, default=None,
                    help="evidence_contract.py to test instead of the installed one")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    m = load_contract(args.module)
    spec = m.PROFILE  # its Tier A fields include the statistic, the wider case
    rng = random.Random(args.seed)
    t0 = time.time()
    failures = []
    for i in range(args.cases):
        bound = gen_bound(m, rng)
        payload = gen_payload(bound, rng)
        try:
            m.verify(payload, bound, spec)  # must not crash on any payload
            cited_clean, claims_clean = m.strip_unsound(payload, bound, spec)
        except Exception as e:  # noqa: BLE001 - a crash is also a finding
            failures.append({"case": i, "error": repr(e), "payload": repr(payload)[:300]})
            if len(failures) >= 5:
                break
            continue
        bad = oracle(bound, cited_clean, claims_clean)
        if bad:
            failures.append({"case": i, "violations": bad[:3], "payload": repr(payload)[:300]})
            if len(failures) >= 5:
                break
    dt = time.time() - t0
    done = (failures[-1]["case"] + 1) if len(failures) >= 5 else args.cases
    result = {"module": str(args.module or "installed"), "cases_run": done, "seed": args.seed,
              "failures": failures, "seconds": round(dt, 1)}
    print(f"{result['module']}: {done:,} cases, {len(failures)} failing, {dt:.0f}s")
    for f in failures[:3]:
        print("  ", json.dumps(f)[:400])
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
