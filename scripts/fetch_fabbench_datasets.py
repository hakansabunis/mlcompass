"""Fetch and normalize the FabBench real-data corpora (Phase 2, R1).

Downloads the four public datasets the frozen task-instance list uses,
normalizes each to a plain comma-separated CSV under ``scripts/data/``, and
prints the SHA256 of every file so run records can pin the exact bytes.

Sources (see scripts/data/README.md for licenses):

  insurance — Medical Cost / Insurance Charges (Lantz, "ML with R")
  heart     — UCI Heart Disease, processed Cleveland subset (clinical)
  telco     — IBM Telco Customer Churn sample
  ames      — Ames Housing (De Cock, JSE 19(3), tab-separated original)

``--verify`` then builds evidence for every frozen (dataset x injector)
instance through the SHIPPED detector and asserts the planted leak is
detected — the zero-API-cost proof that the Phase-2 matrix is runnable
before a single paid call is made.

Usage::

    python scripts/fetch_fabbench_datasets.py            # download missing
    python scripts/fetch_fabbench_datasets.py --force    # re-download all
    python scripts/fetch_fabbench_datasets.py --verify   # check instances
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import urllib.request
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

_HEART_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

_ABALONE_COLUMNS = [
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
    "rings",
]

_AIRFOIL_COLUMNS = [
    "frequency",
    "angle_of_attack",
    "chord_length",
    "free_stream_velocity",
    "suction_thickness",
    "sound_pressure_level",
]

# Candidate URLs are tried in order; all are long-stable public mirrors.
# Entries WITHOUT "extended" form the CORE corpus: the frozen Phase-2 paper
# instances draw only from these (analysis_plan.md §8/A1). Entries WITH
# "extended": True belong to the FabBench artifact's extended corpus — they
# widen the released benchmark's domain coverage (biology, chemistry,
# physics/engineering) without touching the preregistered paper core.
SOURCES: dict[str, dict[str, Any]] = {
    "insurance": {
        "urls": [
            "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv",
        ],
        "out": "insurance.csv",
        "read_kwargs": {},
        "target": "charges",
    },
    "heart": {
        "urls": [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        ],
        "out": "heart_cleveland.csv",
        "read_kwargs": {"header": None, "names": _HEART_COLUMNS, "na_values": "?"},
        "target": "chol",
    },
    "telco": {
        "urls": [
            "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        ],
        "out": "telco_churn.csv",
        "read_kwargs": {},
        "target": "MonthlyCharges",
    },
    "ames": {
        "urls": [
            "https://jse.amstat.org/v19n3/decock/AmesHousing.txt",
            "http://jse.amstat.org/v19n3/decock/AmesHousing.txt",
        ],
        "out": "ames_housing.csv",
        "read_kwargs": {"sep": "\t"},
        "target": "SalePrice",
    },
    "abalone": {
        "urls": [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
        ],
        "out": "abalone.csv",
        "read_kwargs": {"header": None, "names": _ABALONE_COLUMNS},
        "target": "rings",
        "extended": True,
    },
    "wine": {
        "urls": [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        ],
        "out": "wine_quality_red.csv",
        "read_kwargs": {"sep": ";"},
        # NOT the 'quality' score: it is discrete with ~6 levels, and heavy
        # ties push the monotone_log / binned_target correlations below the
        # 0.99 detector threshold (verified 2026-07-07). 'alcohol' is
        # continuous, so all six injectors stay reliably detectable.
        "target": "alcohol",
        "extended": True,
    },
    "airfoil": {
        "urls": [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
        ],
        "out": "airfoil_self_noise.csv",
        "read_kwargs": {"sep": "\t", "header": None, "names": _AIRFOIL_COLUMNS},
        "target": "sound_pressure_level",
        "extended": True,
    },
}

# Real-world CASE STUDIES (analysis_plan.md §8, amendment A2) — natural,
# DOCUMENTED leakage, no injector. Fetched on demand and NOT committed
# (bodyfat: no explicit redistribution license; sambanis: 12 MB, CC0).
#
#   bodyfat  — POSITIVE case: the 'Density' feature deterministically
#              generates the target via Siri's 1956 equation
#              (bodyfat = 495/density - 450); |Spearman| ~ 0.993 >= 0.99,
#              so the shipped detector fires on a leak nobody injected.
#              Documented: Johnson, J. Stat. Educ. 4(1), 1996.
#   sambanis — NEGATIVE control: civil-war onset data whose famous leak
#              (imputation before split; Kapoor & Narayanan, Patterns 2023)
#              is PROCEDURAL — max |feature-target corr| ~ 0.65, no
#              duplicate rows. The detector must stay silent, the contract
#              must abstain (rule 4), and the bare narrator's false-positive
#              fabrication on innocent evidence becomes measurable.
CASE_STUDIES: dict[str, dict[str, Any]] = {
    "bodyfat": {
        "urls": [
            "https://openml.org/data/v1/download/52738/bodyfat.arff",
        ],
        "out": "bodyfat.csv",
        "arff": True,
        "target": "class",
        "kind": "real_leak",
        "leak_column": "Density",
    },
    "sambanis": {
        "urls": [
            "https://dataverse.harvard.edu/api/access/datafile/2701491?format=original",
        ],
        "out": "sambanis_civil_war.csv",
        "read_kwargs": {"low_memory": False},
        "target": "warstds",
        "kind": "negative_control",
    },
}


# The FROZEN Phase-2 instance list (analysis_plan.md §8, amendment A1).
# 12 real-data instances: every injector appears on exactly 2 datasets.
# The synthetic monotone_log task (June 2026) is the 13th, frozen instance.
FROZEN_INSTANCES: list[tuple[str, str]] = [
    ("insurance", "monotone_log"),
    ("insurance", "noisy_proxy"),
    ("insurance", "contamination"),
    ("heart", "exact_copy"),
    ("heart", "inverse_target"),
    ("heart", "binned_target"),
    ("telco", "noisy_proxy"),
    ("telco", "binned_target"),
    ("telco", "contamination"),
    ("ames", "exact_copy"),
    ("ames", "monotone_log"),
    ("ames", "inverse_target"),
]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(urls: list[str]) -> bytes:
    last_error: Exception | None = None
    for url in urls:
        try:
            print(f"  GET {url}", file=sys.stderr)
            req = urllib.request.Request(url, headers={"User-Agent": "fabbench-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return bytes(resp.read())
        except Exception as e:  # noqa: BLE001 — try the next mirror
            last_error = e
            print(f"    failed: {e}", file=sys.stderr)
    raise SystemExit(f"All mirrors failed: {last_error}")


def _normalize(raw: bytes, read_kwargs: dict[str, Any]) -> Any:
    import pandas as pd

    return pd.read_csv(io.BytesIO(raw), **read_kwargs)


def _read_arff(raw: bytes) -> Any:
    """Minimal ARFF reader (numeric attributes, comma-separated @data rows) —
    enough for OpenML's bodyfat file without adding a liac-arff dependency."""
    import pandas as pd

    names: list[str] = []
    data_lines: list[str] = []
    in_data = False
    for line in raw.decode("utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        low = stripped.lower()
        if in_data:
            data_lines.append(stripped)
        elif low.startswith("@attribute"):
            names.append(stripped.split()[1].strip("'\""))
        elif low.startswith("@data"):
            in_data = True
    return pd.read_csv(io.StringIO("\n".join(data_lines)), header=None, names=names)


def fetch_cases(force: bool = False) -> dict[str, str]:
    """Fetch the case-study datasets (on demand; intentionally NOT committed)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, cfg in CASE_STUDIES.items():
        out = os.path.join(DATA_DIR, cfg["out"])
        if os.path.exists(out) and not force:
            print(f"{name} [case]: exists, skipping ({cfg['out']})", file=sys.stderr)
        else:
            raw = _download(cfg["urls"])
            df = (
                _read_arff(raw)
                if cfg.get("arff")
                else _normalize(raw, cfg.get("read_kwargs") or {})
            )
            df.to_csv(out, index=False)
            print(f"{name} [case]: {len(df)} rows x {len(df.columns)} cols -> {cfg['out']}")
        hashes[name] = _sha256(out)
        print(f"  sha256 {cfg['out']}: {hashes[name]}")
    return hashes


def build_case_evidence(name: str) -> dict[str, Any]:
    """Evidence for a real-world case study — NO injector.

    The 'model' is honest about its mechanism: for the positive case it
    exploits the documented leak (Siri's equation on Density); for the
    negative control it is a plain least-squares fit on the numeric features.
    The suspicious-metric value is COMPUTED from those predictions, not
    asserted.
    """
    import numpy as np
    import pandas as pd

    from mlcompass.tools.leakage import detect_leakage

    if name not in CASE_STUDIES:
        raise SystemExit(f"Unknown case study '{name}'. Options: {', '.join(sorted(CASE_STUDIES))}")
    cfg = CASE_STUDIES[name]
    path = os.path.join(DATA_DIR, cfg["out"])
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} missing — run: python scripts/fetch_fabbench_datasets.py --fetch-cases"
        )
    df = pd.read_csv(path, low_memory=False)
    target = cfg["target"]
    y = pd.to_numeric(df[target], errors="coerce")
    df = df.loc[y.notna()].reset_index(drop=True)
    y_arr = y.dropna().to_numpy(dtype=float)

    if cfg["kind"] == "real_leak":
        # Siri (1956): the documented deterministic leak.
        y_pred = 495.0 / df[cfg["leak_column"]].to_numpy(dtype=float) - 450.0
    else:
        features = df.drop(columns=[target]).select_dtypes(include="number")
        x = np.column_stack([features.to_numpy(dtype=float), np.ones(len(df))])
        x = np.nan_to_num(x)
        coef, *_ = np.linalg.lstsq(x, y_arr, rcond=None)
        y_pred = x @ coef
    df["y_pred"] = y_pred

    ss_res = float(np.sum((y_arr - y_pred) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot

    return detect_leakage(
        df,
        y_true_col=target,
        y_pred_col="y_pred",
        task="regression",
        suspicious_metric={"name": "r2", "value": round(r2, 4)},
    )


def verify_cases() -> int:
    """Pre-stated expectations (A2): bodyfat MUST fire on Density with no
    injection; sambanis MUST stay silent (no candidates, low perfect-match)."""
    from mlcompass.agents.leakage_investigator import top_candidate

    failures = 0
    ev = build_case_evidence("bodyfat")
    anchor = top_candidate(ev)
    ok = anchor == "Density"
    r2 = ev.get("suspicious_metric", {}).get("value")
    print(f"bodyfat  [real_leak]        : anchor={anchor} r2={r2} -> {'OK' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    ev = build_case_evidence("sambanis")
    anchor = top_candidate(ev)
    pm = ev.get("perfect_match_rate", 0)
    ok = anchor is None and (pm or 0) < 0.95
    print(
        f"sambanis [negative_control] : candidates={ev.get('candidate_leak_columns')} "
        f"perfect_match={pm} -> {'OK' if ok else 'FAIL'}"
    )
    failures += 0 if ok else 1
    return 1 if failures else 0


def fetch(force: bool = False) -> dict[str, str]:
    os.makedirs(DATA_DIR, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, cfg in SOURCES.items():
        tier = "extended" if cfg.get("extended") else "core"
        out = os.path.join(DATA_DIR, cfg["out"])
        if os.path.exists(out) and not force:
            print(f"{name} [{tier}]: exists, skipping ({cfg['out']})", file=sys.stderr)
        else:
            df = _normalize(_download(cfg["urls"]), cfg.get("read_kwargs") or {})
            df.to_csv(out, index=False)
            print(f"{name} [{tier}]: {len(df)} rows x {len(df.columns)} cols -> {cfg['out']}")
        hashes[name] = _sha256(out)
        print(f"  sha256 {cfg['out']}: {hashes[name]}")
    return hashes


def _check_instance(dataset: str, injector: str) -> tuple[bool, str, int]:
    """Build evidence for one (dataset, injector) cell through the shipped
    detector; return (ok, label, evidence-column count)."""
    from reproduce_hallucination_ablation import build_csv_evidence

    from mlcompass.agents.leakage_investigator import (
        evidence_allowed_columns,
        top_candidate,
    )

    cfg = SOURCES[dataset]
    path = os.path.join(DATA_DIR, cfg["out"])
    if not os.path.exists(path):
        return False, "MISSING FILE — run fetch", 0
    evidence = build_csv_evidence(path, cfg["target"], seed=0, injector=injector)
    anchor = top_candidate(evidence)
    n_cols = len(evidence_allowed_columns(evidence))
    if injector == "contamination":
        ok = evidence.get("perfect_match_rate", 0) >= 0.95 and anchor is None
        label = f"perfect-match {evidence.get('perfect_match_rate'):.3f}"
    else:
        ok = anchor is not None and anchor.endswith("_leak")
        label = anchor or "NOT DETECTED"
    return ok, label, n_cols


def _verify_table(instances: list[tuple[str, str]], title: str) -> int:
    failures = 0
    print(f"\n{title}")
    print("| dataset   | injector       | anchor / channel          | evidence cols |")
    print("| --------- | -------------- | ------------------------- | ------------- |")
    for dataset, injector in instances:
        ok, label, n_cols = _check_instance(dataset, injector)
        if not ok:
            failures += 1
            label += "  <-- FAIL"
        print(f"| {dataset:<9s} | {injector:<14s} | {label:<25s} | {n_cols:>13d} |")
    if failures:
        print(f"\n{failures} instance(s) FAILED verification.")
    else:
        print(f"\nAll {len(instances)} instances verified against the shipped detector.")
    return failures


def verify() -> int:
    """Verify the FROZEN paper instances (analysis_plan.md §8/A1)."""
    from fabbench_injectors import INJECTORS

    assert set(i for _, i in FROZEN_INSTANCES) == set(INJECTORS), (
        "instances must cover all injectors"
    )
    return 1 if _verify_table(FROZEN_INSTANCES, "FROZEN paper instances:") else 0


def verify_extended() -> int:
    """Verify the FabBench extended corpus: every injector on every extended
    dataset (artifact breadth; NOT part of the preregistered paper core)."""
    from fabbench_injectors import list_injectors

    extended = [name for name, cfg in SOURCES.items() if cfg.get("extended")]
    instances = [(d, i) for d in extended for i in list_injectors()]
    return 1 if _verify_table(instances, "EXTENDED corpus (artifact breadth):") else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch/verify the FabBench datasets.")
    ap.add_argument("--force", action="store_true", help="Re-download even if present.")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Build evidence for every frozen instance through the shipped detector.",
    )
    ap.add_argument(
        "--verify-extended",
        action="store_true",
        help="Verify every injector on every extended-corpus dataset (artifact breadth).",
    )
    ap.add_argument(
        "--fetch-cases",
        action="store_true",
        help="Fetch the real-world case-study datasets (bodyfat, sambanis; not committed).",
    )
    ap.add_argument(
        "--verify-cases",
        action="store_true",
        help="Check the A2 case-study expectations (bodyfat fires; sambanis stays silent).",
    )
    args = ap.parse_args()
    if args.fetch_cases:
        fetch_cases(force=args.force)
        return 0
    if args.verify or args.verify_extended or args.verify_cases:
        rc = 0
        if args.verify:
            rc |= verify()
        if args.verify_extended:
            rc |= verify_extended()
        if args.verify_cases:
            rc |= verify_cases()
        return rc
    fetch(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
