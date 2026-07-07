"""Independent re-scorer for manuscript tables (analysis_plan.md A3.9a).

Recomputes the three verification channels (entity / value / omission), the
abstention rate, and the transport-error exclusions from RAW run logs plus
the evidence dict — WITHOUT importing the product's helpers
(``evidence_correlation_map`` / ``top_candidate`` / ``VALUE_TOLERANCE``) or
the measurement harness. Everything below is a second implementation of the
PREREGISTERED rules, so agreement between this scorer and the harness's
inline flags is evidence about the scorer pair, not a tautology. Manuscript
tables are built from THIS scorer (see ``make_tables.py``).

Scoring rules (restated from analysis_plan.md §5, frozen):

- allowed columns  = candidate_leak_columns  UNION  every feature in
  target_feature_correlations (sorted).
- correlation map  = {feature -> correlation} from target_feature_correlations.
- anchor           = candidate_leak_columns[0], or None (anchor-free shapes:
  contamination, sambanis) — omission is UNDEFINED there and excluded from
  omission denominators (A3.3).
- entity violation = any name outside the allowed set, whether it appears in
  columns_referenced OR as a claim's column (an off-evidence claim column is
  a fabricated ENTITY; the value channel is for wrong numbers).
- value violation  = any structured claim whose column IS in the map but
  whose value is not numeric or differs from the measured one by more than
  TOLERANCE = 0.005.
- omission         = a SUBSTANTIVE committed answer (at least one citation
  or claim, verdict neither empty nor cannot_determine) that addresses the
  anchor in neither columns_referenced nor any claim column. A record's
  ``omitted`` flag, when present, is the contract's user-facing post-strip
  output and is read as data (Prop. 2: omissions are flagged, not stripped);
  open arms carry no flag and are recomputed from fields.
- abstention       = verdict == cannot_determine (A3.12).
- transport errors = records with a truthy ``error`` field are NOT data:
  excluded from every denominator, counted per cell (A3.9b).

Usage:
    python scripts/independent_scorer.py --runs-dir scripts/runs
    python scripts/independent_scorer.py --log runs/x.jsonl --evidence runs/evidence_ab12cd34.json
    python scripts/independent_scorer.py --runs-dir scripts/runs --json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from typing import Any

TOLERANCE = 0.005  # restated from the preregistered spec — deliberately NOT imported
_Z95 = 1.959963984540054  # 95% two-sided normal quantile

ABSTAIN_VERDICT = "cannot_determine"

_EV_HASH_RE = re.compile(r"_e([0-9a-fA-F]{6,64})\.jsonl$")


# --------------------------------------------------------------------------- #
# Evidence-side rules                                                          #
# --------------------------------------------------------------------------- #


def allowed_columns(evidence: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for col in evidence.get("candidate_leak_columns") or []:
        if col is not None:
            out.add(str(col))
    for entry in evidence.get("target_feature_correlations") or []:
        if isinstance(entry, dict) and entry.get("feature") is not None:
            out.add(str(entry["feature"]))
    return out


def correlation_map(evidence: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for entry in evidence.get("target_feature_correlations") or []:
        if not isinstance(entry, dict):
            continue
        feature, corr = entry.get("feature"), entry.get("correlation")
        if feature is not None and isinstance(corr, (int, float)):
            out[str(feature)] = float(corr)
    return out


def anchor_column(evidence: dict[str, Any]) -> str | None:
    candidates = evidence.get("candidate_leak_columns") or []
    return str(candidates[0]) if candidates else None


# --------------------------------------------------------------------------- #
# Record-level scoring                                                         #
# --------------------------------------------------------------------------- #


def score_record(
    record: dict[str, Any],
    allowed: set[str],
    corr: dict[str, float],
    anchor: str | None,
) -> dict[str, Any]:
    """Recompute the channel flags for ONE record. ``error`` records are
    marked excluded and carry no channel flags."""
    if record.get("error"):
        return {"excluded_error": True}

    cited = [str(c) for c in (record.get("columns") or [])]
    claims = [c for c in (record.get("claims") or []) if isinstance(c, dict)]
    claim_cols = [str(c.get("column", "")) for c in claims]
    verdict_raw = record.get("verdict")

    entity = any(c not in allowed for c in cited + claim_cols)

    value = False
    for claim in claims:
        col = str(claim.get("column", ""))
        val = claim.get("value")
        if col in corr and (
            not isinstance(val, (int, float)) or abs(float(val) - corr[col]) > TOLERANCE
        ):
            value = True

    omission: bool | None
    if anchor is None:
        omission = None  # undefined on anchor-free evidence (A3.3)
    elif record.get("omitted") is not None:
        # The contract's user-facing flag is part of the response payload
        # (itself re-derived post-strip); read as recorded data.
        omission = bool(record["omitted"])
    else:
        committed = bool(cited or claims) and verdict_raw not in ("", "cannot_determine")
        omission = committed and anchor not in set(cited) | set(claim_cols)

    return {
        "excluded_error": False,
        "entity": entity,
        "value": value,
        "omission": omission,
        "abstained": str(verdict_raw or "") == ABSTAIN_VERDICT,
    }


def wilson(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    z2 = _Z95 * _Z95
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (_Z95 / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


# --------------------------------------------------------------------------- #
# Cell-level aggregation + cross-check against the harness's inline flags      #
# --------------------------------------------------------------------------- #


def score_cell(
    records: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    allowed = allowed_columns(evidence)
    corr = correlation_map(evidence)
    anchor = anchor_column(evidence)

    scored = [score_record(r, allowed, corr, anchor) for r in records]
    valid = [s for s in scored if not s["excluded_error"]]
    n = len(valid)

    def _channel(name: str) -> dict[str, Any]:
        flags = [s[name] for s in valid if s.get(name) is not None]
        k = sum(1 for f in flags if f)
        m = len(flags)
        lo, hi = wilson(k, m)
        return {"k": k, "n": m, "rate": (k / m if m else 0.0), "ci95": [lo, hi]}

    # Cross-check: where the harness stored its own inline flags, count
    # disagreements per channel (scorer-agreement statistic for the paper).
    agreement = {"compared": 0, "disagreements": {"entity": 0, "value": 0, "omission": 0}}
    for r, s in zip(records, scored, strict=True):
        stored = r.get("scored")
        if s["excluded_error"] or not isinstance(stored, dict):
            continue
        agreement["compared"] += 1
        for ch in ("entity", "value"):
            if ch in stored and bool(stored[ch]) != bool(s[ch]):
                agreement["disagreements"][ch] += 1
        if (
            "omission" in stored
            and s["omission"] is not None
            and bool(stored["omission"]) != bool(s["omission"])
        ):
            agreement["disagreements"]["omission"] += 1

    rejections = [
        int(r.get("rejections") or 0)
        for r, s in zip(records, scored, strict=True)
        if not s["excluded_error"]
    ]
    return {
        "n_total": len(records),
        "errors_excluded": len(records) - n,
        "n": n,
        "anchor": anchor,
        "entity": _channel("entity"),
        "value": _channel("value"),
        "omission": (_channel("omission") if anchor is not None else None),
        "abstained": _channel("abstained"),
        "tier_b": {
            "responses_with_catches": sum(1 for x in rejections if x > 0),
            "total_catches": sum(rejections),
        },
        "agreement": agreement,
    }


# --------------------------------------------------------------------------- #
# File plumbing                                                                #
# --------------------------------------------------------------------------- #


def read_jsonl(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break  # truncated crash tail — same policy as the harness
    return out


def load_evidence(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    # Accept both the harness dump ({"task_label", "evidence"}) and a raw dict.
    return payload["evidence"] if isinstance(payload, dict) and "evidence" in payload else payload


def find_evidence_for(log_path: str) -> str | None:
    """Pair a run log with its evidence dump via the content hash embedded in
    the log filename (``..._e<hash>.jsonl`` <-> ``evidence_<hash>.json``)."""
    m = _EV_HASH_RE.search(os.path.basename(log_path))
    if not m:
        return None
    candidate = os.path.join(os.path.dirname(log_path), f"evidence_{m.group(1)}.json")
    return candidate if os.path.exists(candidate) else None


def cell_identity(records: list[dict[str, Any]], log_path: str) -> dict[str, str]:
    """Cell identity from the DATA (records carry provider/task/arm since the
    wave-3 harness); filename only as a fallback label for older logs."""
    first = records[0] if records else {}
    return {
        "provider": str(first.get("provider", "?")),
        "task": str(first.get("task", "?")),
        "model": str(first.get("model", "?")),
        "arm": str(first.get("arm", os.path.basename(log_path))),
        "file": os.path.basename(log_path),
    }


def score_log_file(log_path: str, evidence_path: str | None = None) -> dict[str, Any]:
    records = read_jsonl(log_path)
    ev_path = evidence_path or find_evidence_for(log_path)
    if ev_path is None:
        raise SystemExit(
            f"No evidence file for {log_path}: pass --evidence or keep the "
            "harness's evidence_<hash>.json dump next to the log."
        )
    cell = score_cell(records, load_evidence(ev_path))
    cell["identity"] = cell_identity(records, log_path)
    cell["evidence_file"] = os.path.basename(ev_path)
    return cell


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _fmt_channel(ch: dict[str, Any] | None) -> str:
    if ch is None:
        return "n/a (no anchor)"
    lo, hi = ch["ci95"]
    return f"{ch['k']}/{ch['n']} = {ch['rate'] * 100:.1f}% [{lo * 100:.2f}, {hi * 100:.2f}]"


def main() -> int:
    ap = argparse.ArgumentParser(description="Independent channel re-scorer (A3.9a).")
    ap.add_argument("--log", nargs="*", default=[], help="Run-log JSONL file(s).")
    ap.add_argument("--runs-dir", default=None, help="Score every *.jsonl in this directory.")
    ap.add_argument(
        "--evidence",
        default=None,
        help="Evidence JSON (harness dump or raw dict); default: auto-pair by hash.",
    )
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = ap.parse_args()

    logs = list(args.log)
    if args.runs_dir:
        logs += sorted(
            p
            for p in glob.glob(os.path.join(args.runs_dir, "*.jsonl"))
            if not os.path.basename(p).startswith("evidence_")
        )
    if not logs:
        raise SystemExit("Nothing to score: pass --log file(s) or --runs-dir.")

    results = [score_log_file(p, args.evidence) for p in logs]

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    for cell in results:
        ident = cell["identity"]
        print(f"\n## {ident['provider']} / {ident['model']} / {ident['task']} / {ident['arm']}")
        print(f"   file: {ident['file']}  evidence: {cell['evidence_file']}")
        print(
            f"   N={cell['n']} (of {cell['n_total']}, {cell['errors_excluded']} error(s) excluded)"
        )
        print(f"   entity   : {_fmt_channel(cell['entity'])}")
        print(f"   value    : {_fmt_channel(cell['value'])}")
        print(f"   omission : {_fmt_channel(cell['omission'])}")
        print(f"   abstained: {_fmt_channel(cell['abstained'])}")
        tb = cell["tier_b"]
        print(
            f"   tier B   : {tb['responses_with_catches']} response(s) with catches, "
            f"{tb['total_catches']} total"
        )
        ag = cell["agreement"]
        if ag["compared"]:
            dis = sum(ag["disagreements"].values())
            print(
                f"   cross-check vs harness flags: {ag['compared']} compared, "
                f"{dis} disagreement(s) {ag['disagreements'] if dis else ''}".rstrip()
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
