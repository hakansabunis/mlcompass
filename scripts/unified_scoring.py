"""One row per response, every channel in its own column. Offline.

The 2026-10-01 EMSE review (M1) found that the paper's channel definitions and
its tables did not line up: the profile verifier files a claim's inadmissible
column under its value message, the leakage scorer under entity, and
Definition 2 put the statistic field under entity soundness. This script
scores every valid response of both tasks once, by the definitions as the
paper now states them, and every table is computed from its output.

Definitions (paper, Section 3 and Measures):

  C1  entity soundness: every name in an entity field -- the cited columns and
      each claim's column -- lies in the admissible column set A_col.
      Artifacts (a null column; several admissible names in one string) are
      counted apart.
  C2  value soundness: every claim whose column is admissible names a pair the
      evidence carries ("pair not carried" otherwise, split by whether its
      number is one E carries anywhere) and quotes it within tau ("wrong
      number" otherwise). The statistic field's admissibility is part of C2.
  C3  anchor coverage: a response that commits (a verdict that is not empty
      and not the abstention) references the anchor.

Kinds of violation (2026-10-02 review): every response that breaks the
contract is given one kind, the first that applies.

  extrinsic    it introduces something E does not contain: a name absent from
               E (``unlisted`` when it is a real column of the frame E was
               computed from, ``invented`` otherwise) or a number that appears
               nowhere in E.
  contradicts  it quotes, for a pair E carries, a value E records for
               something else (or a non-number).
  misplaced    it uses only items E contains, in a place the contract does not
               admit: a misfiled name, or a number E carries under a pair E
               does not carry.
  omission     it commits to a verdict without referencing the anchor, and
               breaks nothing else.

The kinds are what an automatic check can decide. Whether a misplaced item
misleads a reader, or states a false relation, is a semantic judgement the
check does not make; the blind audit (analysis plan A7) is there for that.
  The measurement's value table for leakage accepts the correlation aliases
  the first scorer accepts (_CORRELATION_ALIASES) and the dataset-level
  quantities entered against every column; the profile's is the binder's.

    python -X utf8 scripts/unified_scoring.py [--csv benchmark/unified_scoring.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import reproduce_hallucination_ablation as h  # noqa: E402
from analyze_revision_runs import _evidence, _names, _numbers, _records  # noqa: E402

from mlcompass.agents.evidence_contract import PROFILE_VERDICTS, LEAKAGE_VERDICTS, bind_profile  # noqa: E402

RUNS = ROOT / "scripts" / "runs"
TAU = 0.005
VERIFIED = {"layer3", "layer3_bare", "layer3_stress", "layer3_stress_generic", "stress_mech",
            "stress_mech+strict", "strong_verify", "contract", "stress",
            "guardrails_tierb", "guardrails_choices", "guardrails_stock"}
FIELDS = ["run_dir", "file", "task", "names", "provider", "model", "arm", "arm_id", "date",
          "i", "verified", "verdict_admissible", "committed", "c1_cited", "c1_claim",
          "c1_misfiled", "c1_outside_E", "artifact", "c2_wrong", "c2_not_carried",
          "c2_not_carried_real", "c2_unsupported", "c2_wrong_foreign", "c2_wrong_absent",
          "c2_wrong_nonnumeric", "names_unlisted", "names_invented", "c3", "any", "kind",
          "claims", "claims_checked", "claims_exact", "claims_round", "claims_correct",
          "anchor_value", "retried", "retry_reasons"]
_FRAMES: dict[str, set[str]] = {}


def frame_columns(task: str, ctx: dict) -> set[str]:
    """Columns of the frame E was computed from, which separate an unlisted name
    (a real column E does not list) from an invented one."""
    if task == "profile":
        return ctx["cols"]  # the profile admits every column of its frame
    if task in _FRAMES:
        return _FRAMES[task]
    import pandas as pd  # noqa: PLC0415

    if task == "synthetic":  # build_synthetic_evidence
        cols = {"y_true", "log_target_v2", "near_target_proxy", "y_pred"}
        cols |= {f"feature_{i}" for i in range(3, 13)}
    elif task == "synthetic_crowded":
        from fabbench_injectors import crowded_frame  # noqa: PLC0415

        cols = set(map(str, crowded_frame(seed=0)[0].columns))
    elif task.startswith("case:"):
        from fetch_fabbench_datasets import CASE_STUDIES, DATA_DIR  # noqa: PLC0415

        path = pathlib.Path(DATA_DIR) / CASE_STUDIES[task.split(":", 1)[1]]["out"]
        cols = set(pd.read_csv(path, nrows=0, low_memory=False).columns.astype(str)) | {"y_pred"}
    elif task.startswith("csv:"):
        path = ROOT / "scripts" / "data" / task.split(":", 1)[1].split("/", 1)[0]
        cols = set(pd.read_csv(path, nrows=0, low_memory=False).columns.astype(str)) | {"y_pred"}
    else:
        raise ValueError(f"no frame for task {task!r}")
    _FRAMES[task] = cols
    return cols


# The registered follow-up runs of analysis plan A9 (2026-10-02) are reported
# on their own (scripts/analyze_a9.py) and kept out of the corpus counts, which
# describe the runs made up to 2026-09-30.
A9_DIRS = ("2026-10-02_interleaved", "2026-10-02_retry_audit", "2026-10-02_a11",
           "2026-10-03_a14")  # + A11 and A14


def run_files(include_a9: bool = False):
    for f in sorted(RUNS.rglob("*.jsonl")):
        rel = f.relative_to(RUNS).parts
        if any("superseded" in p or p == "transport_errors" for p in rel):
            continue
        if not include_a9 and any(p in A9_DIRS for p in rel):
            continue
        dump = f.parent / f"evidence_{f.stem.rsplit('_e', 1)[1]}.json"
        if dump.exists():
            yield f, rel, _evidence(dump)


def context(kind: str, ev: dict) -> dict:
    # "numbers" is every number E carries: the raw numbers of the dump and the
    # values of the table the contract binds, which derives entries such as the
    # absolute correlation (fault 17: the raw numbers alone missed those).
    if kind == "profile":
        b = bind_profile(ev)
        values = dict(b.values)
        return {"cols": set(b.admissible("columns_referenced")) | set(b.admissible("claims[].column")),
                "values": values, "anchor": b.anchor, "names": _names(ev),
                "numbers": _numbers(ev) + [float(v) for v in values.values()],
                "raw_numbers": _numbers(ev), "norm": lambda s: str(s or ""),
                "verdicts": PROFILE_VERDICTS}
    values = h.evidence_value_table(ev)
    return {"cols": set(h.evidence_allowed_columns(ev)), "values": values,
            "anchor": h.top_candidate(ev), "names": _names(ev),
            "numbers": _numbers(ev) + [float(v) for v in values.values()],
            "raw_numbers": _numbers(ev), "norm": h._normalise_statistic,
            "verdicts": LEAKAGE_VERDICTS}


def score(r: dict, kind: str, ctx: dict, frame: set[str] | None = None) -> dict:
    claims = [c for c in (r.get("raw_claims") if kind == "profile" and not r.get("arm") in VERIFIED
                          else r.get("claims")) or [] if isinstance(c, dict)]
    cited = [str(c) for c in ((r.get("raw_columns") if kind == "profile" and r.get("arm") not in VERIFIED
                               else r.get("columns")) or [])]
    cols = ctx["cols"]
    bad_cited = [c for c in cited if c not in cols and not h.is_scorer_artifact(c, cols)]
    claim_cols = ["None" if c.get("column") is None else str(c.get("column")) for c in claims]
    bad_claim = [c for c in claim_cols if c not in cols and not h.is_scorer_artifact(c, cols)]
    artifact = any(h.is_scorer_artifact(c, cols) for c in cited + claim_cols
                   if c not in cols)
    wrong = not_carried = not_carried_real = unsupported = checked = exact = rounded = 0
    wrong_foreign = wrong_absent = wrong_nonnumeric = 0
    correct_cols: list[str] = []
    for c, col in zip(claims, claim_cols):
        if col not in cols:
            continue
        key = (col, ctx["norm"](c.get("statistic")))
        val = c.get("value")
        num = isinstance(val, (int, float)) and not isinstance(val, bool)
        if key not in ctx["values"]:
            not_carried += 1
            if num and any(abs(float(val) - x) <= TAU for x in ctx["numbers"]):
                not_carried_real += 1
            elif num:
                unsupported += 1
            continue
        if not num:
            wrong += 1
            wrong_nonnumeric += 1
            continue
        d = abs(float(val) - ctx["values"][key])
        checked += 1
        exact += d == 0.0
        rounded += 0.0 < d <= 5e-4
        if d <= TAU:
            correct_cols.append(col)
        if d > TAU:
            wrong += 1
            if any(abs(float(val) - x) <= TAU for x in ctx["numbers"]):
                wrong_foreign += 1
            else:
                wrong_absent += 1
    verdict = str(r.get("verdict") or "").strip()
    committed = bool(verdict) and verdict.lower() not in ("cannot_determine", "cannot determine")
    c3 = committed and ctx["anchor"] is not None and ctx["anchor"] not in set(cited) | set(claim_cols)
    names = ctx["names"]
    out_names = bad_cited + bad_claim
    row = {
        "verdict_admissible": verdict in ctx["verdicts"], "committed": committed,
        "c1_cited": len(bad_cited), "c1_claim": len(bad_claim),
        "c1_misfiled": any(c in names for c in out_names),
        "c1_outside_E": any(c not in names for c in out_names),
        "artifact": artifact, "c2_wrong": wrong, "c2_not_carried": not_carried,
        "c2_not_carried_real": not_carried_real, "c2_unsupported": unsupported,
        "c2_wrong_foreign": wrong_foreign, "c2_wrong_absent": wrong_absent,
        "c2_wrong_nonnumeric": wrong_nonnumeric, "c3": c3,
        "claims": len(claims), "claims_checked": checked, "claims_exact": exact,
        "claims_round": rounded, "claims_correct": len(correct_cols),
        "anchor_value": ctx["anchor"] is not None and ctx["anchor"] in correct_cols,
    }
    row["any"] = bool(bad_cited or bad_claim or wrong or not_carried or c3)
    frame = ctx["cols"] if frame is None else frame
    absent = [c for c in out_names if c not in names]
    row["names_unlisted"] = any(c in frame for c in absent)
    row["names_invented"] = any(c not in frame for c in absent)
    if absent or wrong_absent or unsupported:
        row["kind"] = "extrinsic"
    elif wrong_foreign or wrong_nonnumeric:
        row["kind"] = "contradicts"
    elif bad_cited or bad_claim or not_carried:
        row["kind"] = "misplaced"
    elif c3:
        row["kind"] = "omission"
    else:
        row["kind"] = ""
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--summary", default=None, help="JSON of the counts the paper quotes")
    ap.add_argument("--profile-table", default=None, help="LaTeX rows of the profile table")
    ap.add_argument("--layers-table", default=None, help="LaTeX rows of the two-layer table")
    args = ap.parse_args()
    rows = []
    for f, rel, ev in run_files():
        kind = "profile" if rel[0] == "profile" else "leakage"
        ctx = context(kind, ev)
        for r in _records(f):
            prov = r.get("provenance") or {}
            date = (prov.get("timestamp") or prov.get("utc") or "")[:10] if isinstance(prov, dict) else ""
            row = {"run_dir": "/".join(rel[:-1]) or ".", "file": f.name, "task": r.get("task"),
                   "names": r.get("frame_names") or "", "provider": r.get("provider"),
                   "model": r.get("model"), "arm": r.get("arm"), "arm_id": r.get("arm_id"),
                   "date": date, "i": r.get("i"), "verified": r.get("arm") in VERIFIED,
                   "retried": int(r.get("rejections") or 0) > 0,
                   "retry_reasons": "|".join(str(k) for k in (r.get("rejection_kinds") or [])),
                   **score(r, kind, ctx, frame_columns(str(r.get("task")), ctx))}
            rows.append(row)
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k) for k in FIELDS})
    cells = defaultdict(lambda: defaultdict(int))
    for row in rows:
        key = (row["run_dir"], row["task"], row["names"], row["provider"], row["arm_id"])
        c = cells[key]
        c["n"] += 1
        for k in ("any", "c3", "artifact", "retried", "c1_misfiled", "c1_outside_E"):
            c[k] += bool(row[k])
        c["c1"] += bool(row["c1_cited"] or row["c1_claim"])
        c["c1_claim_only"] += bool(row["c1_claim"] and not row["c1_cited"])
        c["wrong"] += bool(row["c2_wrong"])
        c["not_carried"] += bool(row["c2_not_carried"])
        c["claims_checked"] += row["claims_checked"]
    print(f"{len(rows)} responses in {len(cells)} cells")
    for key, c in sorted(cells.items(), key=lambda kv: str(kv[0])):
        print(f"{str(key)[:100]:100s} n={c['n']:4d} any={c['any']:4d} C1={c['c1']:4d} "
              f"(claim-only {c['c1_claim_only']}) wrong={c['wrong']:3d} notcarried={c['not_carried']:3d} "
              f"C3={c['c3']:3d} retried={c['retried']:3d}")
    summary = summarise(rows)
    if args.summary:
        pathlib.Path(args.summary).write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"wrote {args.summary}")
    if args.profile_table:
        pathlib.Path(args.profile_table).write_text(profile_table(rows), encoding="utf-8")
        print(f"wrote {args.profile_table}")
    if args.layers_table:
        pathlib.Path(args.layers_table).write_text(layers_table(rows), encoding="utf-8")
        print(f"wrote {args.layers_table}")
    return 0


def summarise(rows: list[dict]) -> dict:
    """The counts the paper quotes, each with the universe it is taken over."""
    unverified = [r for r in rows if not r["verified"]]
    tau = {
        "universe": "every response of an arm without a verifier, both tasks, all providers",
        "responses": len(unverified),
        "claims_on_carried_pairs": sum(r["claims_checked"] for r in unverified),
        "exact": sum(r["claims_exact"] for r in unverified),
        "rounded_le_5e-4": sum(r["claims_round"] for r in unverified),
        "responses_with_wrong_number": sum(bool(r["c2_wrong"]) for r in unverified),
    }
    flips = sum(1 for r in rows if r["artifact"] and not r["any"])
    retries: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["retried"]:
            kinds = set(r["retry_reasons"].split("|")) if r["retry_reasons"] else {"unrecorded"}
            label = "malformed only" if kinds == {"malformed"} else (
                "content" if "malformed" not in kinds else "content and malformed")
            retries[f"{r['run_dir']}|{r['task']}|{r['names']}|{r['arm_id']}"][label] += 1
    return {
        "responses": len(rows),
        "by_task": {k: sum(1 for r in rows if (r["task"] == "profile") == (k == "profile"))
                    for k in ("leakage", "profile")},
        "tau": tau,
        "artifacts": {"responses_with_artifact": sum(bool(r["artifact"]) for r in rows),
                      "would_flip_if_counted": flips, "universe": len(rows)},
        "retry_reasons": {k: dict(v) for k, v in retries.items()},
        "layers": {label: counts for label, counts in layer_counts(rows)},
    }


_LEAK_BARE = {"A-L1", "A-L1-T0", "A-GR-STOCK", "A-STRICT-STATIC"}
_LEAK_SPECIFIED = {"A-L1-DESCRIBED", "A-L1-DESCRIBED-NEUTRAL", "A-L2", "A-L2-DESCRIBED",
                   "A-L2-DESCRIBED-NEUTRAL", "A-L2-NOCLAIMS", "A-L2-RULES", "A-STRONG-NOVERIFY"}
_LEAK_CONTRACT = {"A-CONTRACT", "A-L3-SHIPPED", "A-L3-WORST-PARAPHRASE", "A-STRESS",
                  "A-STRESS-GENERIC", "A-STRONG-CONTRACT"}
_PROFILE_CONTRACT = {"P-CONTRACT", "P-STRESS", "P-STRONG-CONTRACT"}


def _leak(r: dict) -> bool:
    return r["task"] != "profile" and r["provider"] == "deepseek"


def _prof(r: dict) -> bool:
    return r["task"] == "profile" and r["provider"] == "deepseek"


LAYER_ROWS = [  # (label, predicate); every response falls in exactly one
    ("Leakage, bare instruction, four instances",
     lambda r: _leak(r) and r["arm_id"] in _LEAK_BARE and "fabbench12" not in r["run_dir"]),
    ("Leakage, bare instruction, twelve instances",
     lambda r: _leak(r) and r["arm_id"] == "A-L1" and "fabbench12" in r["run_dir"]),
    ("Leakage, six instruction wordings",
     lambda r: _leak(r) and str(r["arm_id"]).startswith("A-SWEEP-")),
    ("Leakage, field described or shipped prompt",
     lambda r: _leak(r) and r["arm_id"] in _LEAK_SPECIFIED),
    ("Leakage, enum bound to $E$, no verifier",
     lambda r: _leak(r) and str(r["arm_id"]).startswith("A-STRICT-ENUM")),
    ("Leakage, enum written in advance",
     lambda r: _leak(r) and r["arm_id"] == "A-STATIC-ENUM-STALE"),
    ("Leakage, Guardrails with choices or our checks",
     lambda r: _leak(r) and r["arm_id"] in {"A-GR-CHOICES", "A-GR-OURS"}),
    ("Leakage, contract arms", lambda r: _leak(r) and r["arm_id"] in _LEAK_CONTRACT),
    ("Profile, suffixed names, no verifier",
     lambda r: _prof(r) and r["arm_id"] not in _PROFILE_CONTRACT and r["names"] != "natural"),
    ("Profile, natural names, no verifier",
     lambda r: _prof(r) and r["arm_id"] not in _PROFILE_CONTRACT and r["names"] == "natural"),
    ("Profile, contract arms", lambda r: _prof(r) and r["arm_id"] in _PROFILE_CONTRACT),
    (r"\texttt{gpt-5.4-mini}, every arm", lambda r: r["provider"] == "openai"),
    (r"\texttt{qwen2.5:7b}, every arm", lambda r: r["provider"] == "ollama"),
]


def layer_counts(rows: list[dict]) -> list[tuple[str, dict[str, int]]]:
    """Per group: responses that break the contract, split by kind (the kinds
    partition them), with the extrinsic kind's composition."""
    seen = [0] * len(rows)
    out = []
    for label, pred in LAYER_ROWS:
        cell = []
        for i, r in enumerate(rows):
            if pred(r):
                seen[i] += 1
                cell.append(r)
        counts = {"n": len(cell), "contract": sum(bool(r["any"]) for r in cell)}
        for k in ("extrinsic", "contradicts", "misplaced", "omission"):
            counts[k] = sum(r["kind"] == k for r in cell)
        assert sum(counts[k] for k in ("extrinsic", "contradicts", "misplaced", "omission")) \
            == counts["contract"], label
        counts["invented"] = sum(bool(r["names_invented"]) for r in cell)
        counts["unlisted"] = sum(bool(r["names_unlisted"]) for r in cell)
        counts["number_absent"] = sum(bool(r["c2_wrong_absent"] or r["c2_unsupported"])
                                      for r in cell)
        out.append((label, counts))
    assert all(k == 1 for k in seen), f"{sum(k != 1 for k in seen)} responses not in exactly one group"
    return out


def layers_table(rows: list[dict]) -> str:
    def num(k: int) -> str:
        return f"{k:,}".replace(",", "{,}")

    body = []
    for label, c in layer_counts(rows):
        cells = [num(c[k]) for k in ("n", "contract", "extrinsic", "contradicts",
                                     "misplaced", "omission", "invented")]
        body.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(body) + "\n"


PROFILE_ROWS = [  # (run_dir, names, arm_id, label)
    ("profile", "", "P-L1", "suffixed"),
    ("profile", "", "P-TIER-A-ONLY", "suffixed"),
    ("profile", "", "P-CONTRACT", "suffixed"),
    ("profile", "", "P-STRESS", "suffixed"),
    ("profile/2026-09-24_natural", "natural", "P-L1", "natural"),
    ("profile/2026-09-24_natural", "natural", "P-TIER-A-ONLY", "natural"),
    ("profile/2026-09-30_strong_natural", "natural", "P-STRONG-NOVERIFY", "natural"),
    ("profile/2026-09-30_strong_natural", "natural", "P-STRONG-CONTRACT", "natural"),
    ("profile/2026-09-30_strong", "suffixed", "P-STRONG-NOVERIFY", "suffixed"),
    ("profile/2026-09-30_strong", "suffixed", "P-STRONG-CONTRACT", "suffixed"),
]


def profile_table(rows: list[dict]) -> str:
    from analyze_revision_runs import wilson  # noqa: PLC0415

    body = []
    for run_dir, names, arm_id, label in PROFILE_ROWS:
        cell = [r for r in rows if r["run_dir"] == run_dir and r["arm_id"] == arm_id
                and (r["names"] or "") == names and r["model"] == "deepseek-chat"]
        if run_dir == "profile" and arm_id == "P-CONTRACT":
            cell = [r for r in cell if r["file"] == sorted({x["file"] for x in cell})[0]]
        n = len(cell)
        assert n == 200, (run_dir, arm_id, n)
        c1 = sum(bool(r["c1_cited"] or r["c1_claim"]) for r in cell)
        wrong = sum(bool(r["c2_wrong"]) for r in cell)
        nc = sum(bool(r["c2_not_carried"]) for r in cell)
        anyv = sum(bool(r["any"]) for r in cell)
        lo, hi = wilson(anyv, n)
        rate = "0" if anyv == 0 else f"{100 * anyv / n:.1f} [{lo:.1f}, {hi:.1f}]"
        if any(r["verified"] for r in cell):
            content = sum(1 for r in cell if r["retried"] and r["retry_reasons"].replace("malformed", "").strip("|"))
            malformed = sum(1 for r in cell if r["retried"] and "malformed" in r["retry_reasons"])
            tierb = f"{content}/{malformed}"
        else:
            tierb = "---"
        body.append(f"\\texttt{{{arm_id}}} & {label} & {c1} & {wrong} & {nc} & {rate} & {tierb} \\\\")
    return "\n".join(body) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
