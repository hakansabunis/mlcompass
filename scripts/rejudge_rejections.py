"""Re-judge every stored rejection of the profile contract, independently.

The verifier rejected these attempts and a retry replaced them. The paper's
claim that each rejection was a real violation and that the retry repaired it
rests, for the leakage task, on the verifier's own telemetry, because those
records keep only the kind of each rejection. Profile records written from
2026-09-24 on keep every attempt, so here the claim can be checked without the
verifier: this script reads the evidence dump and the stored attempts and
nothing from the contract code.

For each rejected attempt it lists the items that break the evidence, by its
own reading of E (a column E lists, a (column, statistic) pair E records, the
recorded value within tau), says whose value a wrong number is, and says what
the next attempt did with each item.

    python -X utf8 scripts/rejudge_rejections.py [--tex paper/tse_latex/table_rejections_rows.tex]
"""

from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "scripts" / "runs" / "profile"
LEAK_RUNS = ROOT / "scripts" / "runs"
TAU = 0.005
VERDICTS = {"clean", "needs_cleaning", "unusable", "cannot_determine"}


def evidence_table(ev: dict) -> tuple[set[str], dict[tuple[str, str], float]]:
    """The columns E lists and the value it records for each (column, statistic)."""
    ev = ev.get("evidence", ev)
    cols, values = set(), {}
    for c in ev.get("columns") or []:
        name = str(c.get("name"))
        cols.add(name)
        for k, v in c.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values[(name, k)] = float(v)
        for k, v in (c.get("stats") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values[(name, k)] = float(v)
    return cols, values


def problems(attempt: dict, cols: set[str], values: dict) -> list[tuple[str, str]]:
    """(item, finding) for every item of one attempt that E does not support."""
    out = []
    if str(attempt.get("verdict") or "") not in VERDICTS:
        out.append(("verdict", "no admissible verdict"))
    for c in attempt.get("columns_referenced") or []:
        if str(c) not in cols:
            out.append((str(c), "column E does not list"))
    for cl in attempt.get("claims") or []:
        col, stat, v = str(cl.get("column")), str(cl.get("statistic")), cl.get("value")
        item = f"{col}.{stat}"
        if col not in cols:
            out.append((item, "column E does not list"))
        elif (col, stat) not in values:
            out.append((item, "pair E does not record"))
        elif not isinstance(v, (int, float)) or isinstance(v, bool):
            out.append((item, "value is not a number"))
        elif abs(float(v) - values[(col, stat)]) > TAU:
            owners = [f"{c2}.{s2}" for (c2, s2), x in values.items()
                      if abs(float(v) - x) <= TAU and (c2, s2) != (col, stat)]
            whose = f"; the value of {owners[0]}" if len(owners) == 1 else (
                "; a value E records nowhere" if not owners else f"; a value of {len(owners)} items")
            out.append((item, f"wrong number ({v:g} for {values[(col, stat)]:g}{whose})"))
    return out


def fate(item: str, nxt: dict, cols: set[str], values: dict) -> str:
    """What the next attempt did with an item the previous one got wrong."""
    if item == "verdict":
        return "verdict supplied" if str(nxt.get("verdict") or "") in VERDICTS else "still missing"
    if "." not in item:
        return "dropped" if item not in (nxt.get("columns_referenced") or []) else "kept"
    col, stat = item.split(".", 1)
    same = [cl for cl in nxt.get("claims") or [] if str(cl.get("column")) == col
            and str(cl.get("statistic")) == stat]
    if not same:
        return "dropped"
    v = same[0].get("value")
    if (col, stat) in values and isinstance(v, (int, float)) and abs(float(v) - values[(col, stat)]) <= TAU:
        return "corrected"
    return "kept"


def rejudge() -> tuple[list[dict], int, int]:
    """Every stored rejection, re-judged: (rows, rejected attempts, agreements)."""
    rows, agree, total = [], 0, 0
    for f in sorted(RUNS.rglob("*.jsonl")):
        if any("superseded" in p or p == "transport_errors" for p in f.parts):
            continue
        dump = f.parent / f"evidence_{f.stem.rsplit('_e', 1)[1]}.json"
        cols, values = evidence_table(json.loads(dump.read_text(encoding="utf-8")))
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            at = r.get("attempts") or []
            for a, nxt in zip(at, at[1:]):
                total += 1
                found = problems(a, cols, values)
                agree += bool(found) == bool(a.get("violations") or not a.get("verdict"))
                final_clean = not problems(at[-1], cols, values)
                for item, finding in found or [("--", "nothing found")]:
                    rows.append({"run": f.parent.name, "arm": r.get("arm_id"), "i": r.get("i"),
                                 "item": item, "finding": finding,
                                 "fate": fate(item, nxt, cols, values) if found else "--",
                                 "final_clean": final_clean})
    return rows, total, agree


def judge_leakage_attempt(attempt: dict, ev: dict) -> list[str]:
    """What the second scorer (scripts/definitions_scorer.py, which shares no
    code with the verifier) finds wrong in one leakage attempt."""
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(ROOT / "scripts"))
    import definitions_scorer as ds  # noqa: PLC0415

    verdict = str(attempt.get("verdict") or "")
    if verdict not in ds.COMMITTING | {"cannot_determine"}:
        return ["no admissible verdict"]
    s = ds.score({"columns": attempt.get("columns_referenced") or [],
                  "claims": attempt.get("claims") or [], "verdict": verdict}, ev)
    found = [f"name outside E: {n}" for n in s["outside_names"]]
    if s["wrong_number"]:
        found.append("wrong number")
    if s["omission"]:
        found.append("omission")
    return found


def rejudge_leakage() -> tuple[list[dict], int]:
    """Every rejected leakage attempt that the records keep (harness of
    2026-10-01 or later), re-judged by the second scorer."""
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(ROOT / "scripts"))
    import definitions_scorer as ds  # noqa: PLC0415

    rows, total = [], 0
    for f in sorted(LEAK_RUNS.rglob("*.jsonl")):
        if "profile" in f.parts or any("superseded" in p or p == "transport_errors"
                                       for p in f.parts):
            continue
        dump = f.parent / f"evidence_{f.stem.rsplit('_e', 1)[-1]}.json"
        if not dump.exists():
            continue
        ev = ds.load_evidence(dump)
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            at = r.get("attempts") or []
            for a in at[:-1]:
                total += 1
                found = judge_leakage_attempt(a, ev)
                rows.append({"run": f.parent.name, "arm": r.get("arm_id"), "i": r.get("i"),
                             "found": found, "final_clean": not judge_leakage_attempt(at[-1], ev)})
    return rows, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=None)
    args = ap.parse_args()
    leak_rows, leak_total = rejudge_leakage()
    print(f"leakage: {leak_total} rejected attempts kept; the second scorer finds a problem in "
          f"{sum(bool(r['found']) for r in leak_rows)}; final attempts clean: "
          f"{sum(r['final_clean'] for r in leak_rows)}")
    rows, total, agree = rejudge()
    for row in rows:
        print(row)
    print(f"{total} rejected attempts; independent reading finds a problem in "
          f"{sum(1 for _ in {(r['run'], r['arm'], r['i']) for r in rows if r['finding'] != 'nothing found'})}; "
          f"agrees with the verifier on {agree}")
    if args.tex:
        def tt(s: str) -> str:
            return "\\texttt{" + s.replace("_", "\\_") + "}"

        def short(item: str, finding: str) -> str:
            if finding.startswith("wrong number"):
                owner = finding.split("the value of ", 1)[1].rstrip(")") if "the value of " in finding else ""
                if not owner:
                    return "wrong number"
                o_col, o_stat = owner.split(".", 1)
                return "value of " + tt(o_col if item.endswith("." + o_stat) else owner)
            return {"column E does not list": "column not in $E$",
                    "pair E does not record": "pair not in $E$"}.get(finding, finding)

        # One row per item; an attempt with more than three items of one finding
        # and one fate is shown as one row, with the first two items named.
        groups: dict[tuple, list[str]] = {}
        for row in rows:
            run = row["run"][5:10]  # 2026-09-24_attempts -> 09-24
            k = (run, row["arm"], row["i"], short(row["item"], row["finding"]), row["fate"])
            groups.setdefault(k, []).append(row["item"])
        lines = []
        for (run, arm, i, finding, fate), items in groups.items():
            shown = [tt(x) for x in items] if len(items) <= 3 else [
                f"{tt(items[0])}, {tt(items[1])}, and {len(items) - 2} more"]
            for item in shown:
                lines.append(f"{run} & {tt(arm)} & {i} & {item} & {finding} & {fate} \\\\")
        pathlib.Path(args.tex).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
