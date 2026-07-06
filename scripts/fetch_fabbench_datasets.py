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

# Candidate URLs are tried in order; all are long-stable public mirrors.
SOURCES: dict[str, dict[str, Any]] = {
    "insurance": {
        "urls": [
            "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv",
        ],
        "out": "insurance.csv",
        "format": "csv",
        "target": "charges",
    },
    "heart": {
        "urls": [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        ],
        "out": "heart_cleveland.csv",
        "format": "uci_heart",
        "target": "chol",
    },
    "telco": {
        "urls": [
            "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        ],
        "out": "telco_churn.csv",
        "format": "csv",
        "target": "MonthlyCharges",
    },
    "ames": {
        "urls": [
            "https://jse.amstat.org/v19n3/decock/AmesHousing.txt",
            "http://jse.amstat.org/v19n3/decock/AmesHousing.txt",
        ],
        "out": "ames_housing.csv",
        "format": "tsv",
        "target": "SalePrice",
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


def _normalize(raw: bytes, fmt: str) -> Any:
    import pandas as pd

    if fmt == "csv":
        return pd.read_csv(io.BytesIO(raw))
    if fmt == "tsv":
        return pd.read_csv(io.BytesIO(raw), sep="\t")
    if fmt == "uci_heart":
        return pd.read_csv(io.BytesIO(raw), header=None, names=_HEART_COLUMNS, na_values="?")
    raise SystemExit(f"Unknown format: {fmt}")


def fetch(force: bool = False) -> dict[str, str]:
    os.makedirs(DATA_DIR, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, cfg in SOURCES.items():
        out = os.path.join(DATA_DIR, cfg["out"])
        if os.path.exists(out) and not force:
            print(f"{name}: exists, skipping ({cfg['out']})", file=sys.stderr)
        else:
            df = _normalize(_download(cfg["urls"]), cfg["format"])
            df.to_csv(out, index=False)
            print(f"{name}: {len(df)} rows x {len(df.columns)} cols -> {cfg['out']}")
        hashes[name] = _sha256(out)
        print(f"  sha256 {cfg['out']}: {hashes[name]}")
    return hashes


def verify() -> int:
    """Build evidence for every frozen instance; assert the leak is detected."""
    from fabbench_injectors import INJECTORS
    from reproduce_hallucination_ablation import build_csv_evidence

    from mlcompass.agents.leakage_investigator import (
        evidence_allowed_columns,
        top_candidate,
    )

    failures = 0
    print("\n| dataset   | injector       | anchor / channel        | evidence cols |")
    print("| --------- | -------------- | ------------------------ | ------------- |")
    for dataset, injector in FROZEN_INSTANCES:
        cfg = SOURCES[dataset]
        path = os.path.join(DATA_DIR, cfg["out"])
        if not os.path.exists(path):
            print(f"| {dataset:<9s} | {injector:<14s} | MISSING FILE — run fetch |")
            failures += 1
            continue
        evidence = build_csv_evidence(path, cfg["target"], seed=0, injector=injector)
        anchor = top_candidate(evidence)
        n_cols = len(evidence_allowed_columns(evidence))
        if injector == "contamination":
            ok = evidence.get("perfect_match_rate", 0) >= 0.95 and anchor is None
            label = f"perfect-match {evidence.get('perfect_match_rate'):.3f}"
        else:
            ok = anchor is not None and anchor.endswith("_leak")
            label = anchor or "NOT DETECTED"
        if not ok:
            failures += 1
            label += "  <-- FAIL"
        print(f"| {dataset:<9s} | {injector:<14s} | {label:<24s} | {n_cols:>13d} |")
    assert set(i for _, i in FROZEN_INSTANCES) == set(INJECTORS), (
        "instances must cover all injectors"
    )
    if failures:
        print(f"\n{failures} instance(s) FAILED verification.")
        return 1
    print(f"\nAll {len(FROZEN_INSTANCES)} frozen instances verified against the shipped detector.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch/verify the FabBench datasets.")
    ap.add_argument("--force", action="store_true", help="Re-download even if present.")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Build evidence for every frozen instance through the shipped detector.",
    )
    args = ap.parse_args()
    if args.verify:
        return verify()
    fetch(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
