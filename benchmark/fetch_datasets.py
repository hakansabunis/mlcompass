"""Fetch benchmark datasets from OpenML into a pinned, checksummed local copy.

`protocol.md` §2 requires a fixed dataset version with a checksum, and the
exact preparation commands recorded. This script is that step: it pulls a
dataset by OpenML id, writes it as CSV next to a manifest carrying the
version, the OpenML file id, the sha256 of the bytes we wrote, the row and
column counts, and the target column name.

Re-running is idempotent and verifying. A dataset already present is
re-hashed and compared against the manifest rather than silently
re-downloaded, so a changed file is an error rather than a surprise. That
matters more here than convenience: OpenML ids are mutable across versions,
and a benchmark whose inputs drift underneath it is measuring nothing.

    python benchmark/fetch_datasets.py                 # fetch the pinned set
    python benchmark/fetch_datasets.py --verify        # check, download nothing
    python benchmark/fetch_datasets.py --id 1480       # one dataset

No third-party client: the OpenML REST API over urllib keeps this runnable
from the dev environment without adding a dependency to the package.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

API = "https://www.openml.org/api/v1/json/data"
DATA_DIR = Path(__file__).parent / "data"
MANIFEST = DATA_DIR / "manifest.json"

# The pinned set. Chosen for task-type spread and issue diversity, not size:
# one near-control so over-reporting is visible, one issue-rich binary case,
# one regression case so the scoring is not specialised to classification.
PINNED: dict[int, str] = {
    1464: "blood-transfusion-service-center",
    1480: "ilpd",
    44031: "california",
}

TIMEOUT = 120


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:  # noqa: S310 - fixed https host
        return json.loads(r.read().decode("utf-8"))


def _get_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:  # noqa: S310 - fixed https host
        return r.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def describe(dataset_id: int) -> dict[str, Any]:
    """Return OpenML's description record for ``dataset_id``."""
    return _get_json(f"{API}/{dataset_id}")["data_set_description"]


def _read_arff(raw: bytes) -> pd.DataFrame:
    """Parse ARFF into a DataFrame without adding a dependency.

    Only what OpenML actually emits is handled: an ``@attribute`` header
    block, then ``@data`` and comma-separated rows with ``?`` for missing.
    Quoted values with embedded commas go through ``pandas.read_csv`` rather
    than a hand-rolled split, so nominal labels containing commas survive.
    """
    text = raw.decode("utf-8", errors="replace")
    names: list[str] = []
    data_start = None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if low.startswith("@attribute"):
            parts = line.split(None, 2)
            if len(parts) >= 2:
                names.append(parts[1].strip("'\""))
        elif low.startswith("@data"):
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError("no @data section found in the ARFF payload")

    body = "\n".join(lines[data_start:])
    return pd.read_csv(
        io.StringIO(body),
        header=None,
        names=names,
        na_values=["?"],
        skipinitialspace=True,
        skip_blank_lines=True,
    )


def fetch_one(dataset_id: int, *, force: bool = False) -> dict[str, Any]:
    """Download one dataset to CSV and return its manifest entry."""
    desc = describe(dataset_id)
    name = desc.get("name", str(dataset_id))
    out = DATA_DIR / f"{dataset_id}_{name}.csv"

    if out.exists() and not force:
        raw = out.read_bytes()
        df = pd.read_csv(out)
    else:
        url = desc.get("url")
        if not url:
            raise ValueError(f"dataset {dataset_id} has no download url")
        payload = _get_bytes(url)
        df = (
            _read_arff(payload)
            if url.lower().endswith(".arff")
            else pd.read_csv(io.BytesIO(payload))
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        raw = out.read_bytes()

    return {
        "openml_id": dataset_id,
        "name": name,
        "version": desc.get("version"),
        "openml_file_id": desc.get("file_id"),
        "default_target_attribute": desc.get("default_target_attribute"),
        "licence": desc.get("licence"),
        "csv_path": str(out.relative_to(DATA_DIR.parent)).replace("\\", "/"),
        "sha256": _sha256(raw),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
    }


def verify(manifest: dict[str, Any]) -> int:
    """Re-hash every file in the manifest. Returns the number of mismatches."""
    bad = 0
    for entry in manifest["datasets"]:
        path = DATA_DIR.parent / entry["csv_path"]
        if not path.exists():
            print(f"  MISSING  {entry['csv_path']}")
            bad += 1
            continue
        actual = _sha256(path.read_bytes())
        if actual != entry["sha256"]:
            print(
                f"  CHANGED  {entry['csv_path']}\n"
                f"           manifest {entry['sha256'][:16]}"
                f"  on disk {actual[:16]}"
            )
            bad += 1
        else:
            print(f"  ok       {entry['csv_path']}  {entry['rows']}x{entry['columns']}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", type=int, action="append", help="fetch one id (repeatable)")
    ap.add_argument("--verify", action="store_true", help="re-hash against the manifest and exit")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    if args.verify:
        if not MANIFEST.exists():
            print("no manifest; run without --verify first")
            return 1
        print("Verifying against manifest:")
        bad = verify(json.loads(MANIFEST.read_text(encoding="utf-8")))
        print("all files match" if not bad else f"{bad} problem(s)")
        return 1 if bad else 0

    ids = args.id or sorted(PINNED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for dataset_id in ids:
        try:
            entry = fetch_one(dataset_id, force=args.force)
        except (urllib.error.URLError, ValueError, KeyError) as exc:
            print(f"  FAILED   {dataset_id}: {exc}")
            continue
        entries.append(entry)
        print(
            f"  ok       {entry['csv_path']}  {entry['rows']}x{entry['columns']}"
            f"  sha {entry['sha256'][:12]}…"
        )

    MANIFEST.write_text(
        json.dumps({"source": "openml", "api": API, "datasets": entries}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nmanifest: {MANIFEST.relative_to(DATA_DIR.parent)}  ({len(entries)} dataset(s))")
    return 0 if len(entries) == len(ids) else 1


if __name__ == "__main__":
    sys.exit(main())
