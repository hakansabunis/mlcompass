"""Analysis plan A10: the semantic audit by two NLI judges.

Three steps, each writing a file the next one reads.

  build  For each of the 204 items of the A7 sheet, a premise generated from E
         by fixed templates and a hypothesis generated from the item. Writes
         benchmark/semantic_audit_A7/nli_inputs.json, which carries no stratum.
  judge  Runs the two NLI models on the inputs. Needs torch and transformers,
         which the project environment does not carry; run it with the
         separate environment (for example C:/Users/<you>/nli-venv). Writes
         nli_labels.json.
  score  Joins the labels with the sealed key: per judge and stratum, the
         label shares with Wilson intervals, the validity check on strata of
         known truth, and Cohen's kappa between the judges. Writes
         nli_results.json and the LaTeX rows of the paper's table.

    python -X utf8 scripts/nli_audit.py build
    <nli-venv>/python -X utf8 scripts/nli_audit.py judge
    python -X utf8 scripts/nli_audit.py score [--tex paper/tse_latex/table_nli_rows.tex]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDIT = ROOT / "benchmark" / "semantic_audit_A7"
RUNS = ROOT / "scripts" / "runs"
TAU = 0.005
JUDGES = {
    "deberta": ("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"),
    "roberta": ("FacebookAI/roberta-large-mnli", "2a8f12d27941090092df78e4ba6f0928eb5eac98"),
}
STAT_NAMES = {
    "correlation": "correlation with the target", "corr": "correlation with the target",
    "correlation_with_target": "correlation with the target",
    "pearson": "Pearson correlation with the target",
    "pearson_correlation": "Pearson correlation with the target",
    "spearman": "Spearman correlation with the target",
    "spearman_correlation": "Spearman correlation with the target",
    "abs_corr": "absolute correlation with the target",
    "max_abs_correlation": "absolute correlation with the target",
    "perfect_match_rate": "perfect-match rate", "row_count": "row count",
    "missing_count": "number of missing values", "missing_pct": "missing rate",
    "zero_ratio": "share of zeros", "iqr_count": "number of interquartile-range outliers",
    "z_score_count": "number of z-score outliers", "cardinality": "number of distinct values",
    "mean": "mean", "std": "standard deviation", "min": "minimum", "max": "maximum",
    "q25": "25th percentile", "q50": "median", "q75": "75th percentile",
    "class_balance": "class balance", "rows": "number of rows", "cols": "number of columns",
}
KNOWN_SUPPORTED = {"S4 real column E does not list", "S8 control: admissible name",
                   "S9 control: claim that matches E"}
KNOWN_UNSUPPORTED = {"S3 wrong number, another item's value", "S5 invented name, list written in advance",
                     "S6 invented name", "S7 number E records nowhere"}


def stat_name(s) -> str:
    s = str(s or "")
    return STAT_NAMES.get(s.lower().strip().replace(" ", "_").replace("-", "_"), s.replace("_", " "))


def num(v) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int) or (isinstance(v, float) and math.isfinite(v) and float(v).is_integer()
                              and abs(v) >= 1):
        return str(int(v))
    if isinstance(v, float) and math.isfinite(v):
        return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:.1f}"
    return str(v)


def sentences(ev: dict, frame: list[str]) -> list[tuple[str, str | None, float | None]]:
    """(sentence, the column it is about or None, its number or None)."""
    out: list[tuple[str, str | None, float | None]] = []
    if "columns" in ev and isinstance(ev["columns"], list):           # profile
        sh = ev.get("shape") or {}
        if sh:
            out.append((f"The data has {sh.get('rows')} rows.", None, float(sh.get("rows") or 0)))
            out.append((f"The data has {sh.get('cols')} columns.", None, float(sh.get("cols") or 0)))
        th = ev.get("target_hint") or {}
        if th.get("column"):
            out.append((f"The target column is {th['column']}.", None, None))
        task = ev.get("task_hint") or {}
        if task.get("type"):
            out.append((f"The task is {str(task['type']).replace('_', ' ')}.", None, None))
        cb = task.get("class_balance") or {}
        if cb:
            parts = " and ".join(f"{num(v)} for class {k}" for k, v in cb.items())
            out.append((f"The class balance of the target column {th.get('column', 'target')} is {parts}.",
                        th.get("column"), None))
            for v in cb.values():
                out.append((f"One class of the target makes up a share of {num(v)}.", th.get("column"),
                            float(v)))
        for c in ev["columns"]:
            name = str(c.get("name"))
            out.append((f"Column {name} is {c.get('type')}.", name, None))
            for k, v in c.items():
                if k in ("name", "type", "stats") or not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                out.append((f"The {stat_name(k)} of column {name} is {num(v)}.", name, float(v)))
            for k, v in (c.get("stats") or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.append((f"The {stat_name(k)} of column {name} is {num(v)}.", name, float(v)))
    else:                                                               # leakage
        if "row_count" in ev:
            out.append((f"The data has {ev['row_count']} rows.", None, float(ev["row_count"])))
        sm = ev.get("suspicious_metric") or {}
        if sm:
            out.append((f"The suspicious metric is {sm.get('name')}, with value {num(sm.get('value'))}.",
                        None, None))
        for e in ev.get("target_feature_correlations") or []:
            f, r = str(e.get("feature")), e.get("correlation")
            if isinstance(r, (int, float)):
                out.append((f"The column {f} has a {e.get('method', '')} correlation of {num(r)} with the "
                            "target.".replace("  ", " "), f, float(r)))
                out.append((f"The absolute correlation of column {f} with the target is {num(abs(r))}.",
                            f, abs(float(r))))
        if "perfect_match_rate" in ev and isinstance(ev["perfect_match_rate"], (int, float)):
            out.append((f"The perfect-match rate between the predictions and the target is "
                        f"{num(ev['perfect_match_rate'])}.", None, float(ev["perfect_match_rate"])))
        cands = ev.get("candidate_leak_columns") or []
        out.append((f"The candidate leak columns are {', '.join(map(str, cands))}." if cands
                    else "There are no candidate leak columns.", None, None))
    if frame:
        out.append((f"The data has the columns {', '.join(frame)}.", None, None))
    return out


def premise_for(ev: dict, frame: list[str], column: str | None, value: float | None) -> str:
    sents = sentences(ev, frame)
    own = [s for s, c, _ in sents if column is not None and c == column]
    same = [s for s, c, v in sents if value is not None and v is not None and abs(v - value) <= TAU
            and s not in own]
    general = [s for s, c, _ in sents if c is None]
    seen, out = set(), []
    for s in own + same + general:            # the item's own facts first: truncation cuts the end
        if s not in seen:
            seen.add(s)
            out.append(s)
    return " ".join(out)


def load_sheet() -> list[dict]:
    html = (AUDIT / "audit_sheet.html").read_text(encoding="utf-8")
    m = re.search(r"const ITEMS = (\[.*?\]);\nconst CHOICES", html, flags=re.S)
    return json.loads(m.group(1))


def build() -> int:
    key = json.loads((AUDIT / "KEY_sealed.json").read_text(encoding="utf-8"))
    rows = []
    for it in load_sheet():
        k = key[str(it["id"])]
        rel = pathlib.Path(k["file"].lstrip("/"))  # runs at the top level carry a leading "/"
        dump = RUNS / rel.parent / ("evidence_" + rel.stem.rsplit("_e", 1)[1] + ".json")
        raw = json.loads(dump.read_text(encoding="utf-8"))
        ev = raw.get("evidence", raw)
        frame = list(it["evidence"]["frame"])
        if it["target_type"] == "claim":
            c = it["claims"][int(it["target"])]
            col, v = str(c.get("column")), c.get("value")
            value = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
            hyp = f"The {stat_name(c.get('statistic'))} of column {col} is {num(v)}."
            prem = premise_for(ev, frame, col, value)
        else:
            name = str(it["target"])
            hyp = f"{name} is a column of the data."
            prem = premise_for(ev, frame, name, None)
        rows.append({"id": it["id"], "premise": prem, "hypothesis": hyp})
    out = AUDIT / "nli_inputs.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(rows)} items)")
    return 0


def judge() -> int:
    """Both judges, one item at a time, appended to nli_labels.jsonl so that an
    interrupted run resumes where it stopped; nli_labels.json is written from
    the complete log at the end."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

    rows = json.loads((AUDIT / "nli_inputs.json").read_text(encoding="utf-8"))
    log = AUDIT / "nli_labels.jsonl"
    done = set()
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add((rec["judge"], str(rec["id"])))
    device = "cuda" if torch.cuda.is_available() else "cpu"  # same fp32 weights either way
    print(f"device: {device}", flush=True)
    for short, (repo, rev) in JUDGES.items():
        todo = [r for r in rows if (short, str(r["id"])) not in done]
        if not todo:
            continue
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        model = AutoModelForSequenceClassification.from_pretrained(repo, revision=rev).eval().to(device)
        id2label = {i: lab.lower() for i, lab in model.config.id2label.items()}
        for k, r in enumerate(todo, 1):
            enc = tok(r["premise"], r["hypothesis"], truncation="only_first", max_length=512,
                      return_tensors="pt").to(device)
            with torch.no_grad():
                p = torch.softmax(model(**enc).logits[0], dim=-1).tolist()
            probs = {id2label[i]: round(x, 4) for i, x in enumerate(p)}
            rec = {"judge": short, "id": r["id"], "label": max(probs, key=probs.get), "probs": probs,
                   "truncated": len(tok(r["premise"], r["hypothesis"])["input_ids"]) > 512,
                   "device": device}
            with log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            if k % 20 == 0:
                print(f"{short}: {k}/{len(todo)}", flush=True)
        print(f"{short}: judged {len(todo)} items", flush=True)
    labels: dict[str, dict] = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            labels.setdefault(str(rec["id"]), {})[rec["judge"]] = {
                "label": rec["label"], "probs": rec["probs"], "truncated": rec["truncated"]}
    out = AUDIT / "nli_labels.json"
    out.write_text(json.dumps({"judges": {k: {"repo": v[0], "revision": v[1]} for k, v in JUDGES.items()},
                               "labels": labels}, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.959964, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


READ = {"entailment": "supported", "neutral": "not addressed", "contradiction": "contradicted"}


def score(tex: str | None) -> int:
    key = json.loads((AUDIT / "KEY_sealed.json").read_text(encoding="utf-8"))
    lab = json.loads((AUDIT / "nli_labels.json").read_text(encoding="utf-8"))["labels"]
    res: dict = {"by_stratum": {}, "validity": {}, "kappa": None, "truncated": {}}
    strata = sorted({k["stratum"] for k in key.values()})
    for j in JUDGES:
        by = defaultdict(Counter)
        for i, k in key.items():
            by[k["stratum"]][READ[lab[i][j]["label"]]] += 1
        res["by_stratum"][j] = {s: dict(by[s]) for s in strata}
        sup_n = sum(sum(by[s].values()) for s in KNOWN_SUPPORTED)
        sup_ok = sum(by[s]["supported"] for s in KNOWN_SUPPORTED)
        uns_n = sum(sum(by[s].values()) for s in KNOWN_UNSUPPORTED)
        uns_ok = sum(sum(by[s].values()) - by[s]["supported"] for s in KNOWN_UNSUPPORTED)
        res["validity"][j] = {
            "known_supported": [sup_ok, sup_n, wilson(sup_ok, sup_n)],
            "known_unsupported": [uns_ok, uns_n, wilson(uns_ok, uns_n)],
            "valid": sup_ok >= 0.9 * sup_n and uns_ok >= 0.9 * uns_n}
        res["truncated"][j] = sum(lab[i][j]["truncated"] for i in key)
    # Fault 17: the same check with the drawn items re-assigned by every number E carries.
    import make_semantic_audit as msa  # noqa: PLC0415

    fixed = msa.corrected_strata(key)
    res["moved_by_fault_17"] = dict(Counter(f"{key[i]['stratum']} -> {fixed[i]}"
                                            for i in key if fixed[i] != key[i]["stratum"]))
    res["validity_corrected"] = {}
    for j in JUDGES:
        sup = [i for i in key if fixed[i] in KNOWN_SUPPORTED]
        uns = [i for i in key if fixed[i] in KNOWN_UNSUPPORTED]
        sup_ok = sum(READ[lab[i][j]["label"]] == "supported" for i in sup)
        uns_ok = sum(READ[lab[i][j]["label"]] != "supported" for i in uns)
        res["validity_corrected"][j] = {
            "known_supported": [sup_ok, len(sup), wilson(sup_ok, len(sup))],
            "known_unsupported": [uns_ok, len(uns), wilson(uns_ok, len(uns))],
            "valid": sup_ok >= 0.9 * len(sup) and uns_ok >= 0.9 * len(uns)}
    ids = sorted(key)
    res["kappa"] = kappa([lab[i]["deberta"]["label"] for i in ids], [lab[i]["roberta"]["label"] for i in ids])
    res["agreement"] = sum(lab[i]["deberta"]["label"] == lab[i]["roberta"]["label"] for i in ids) / len(ids)
    out = AUDIT / "nli_results.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    if tex:
        lines = []
        for s in strata:
            n = sum(res["by_stratum"]["deberta"][s].values())
            cells = []
            for j in JUDGES:
                c = res["by_stratum"][j][s]
                cells += [str(c.get("supported", 0)), str(c.get("not addressed", 0)),
                          str(c.get("contradicted", 0))]
            label = s.split(" ", 1)[1].replace("E", "$E$").replace("_", r"\_")
            lines.append(f"{s.split(' ', 1)[0]} {label} & {n} & " + " & ".join(cells) + r" \\")
        pathlib.Path(tex).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {tex}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "judge", "score"])
    ap.add_argument("--tex", default=None)
    a = ap.parse_args()
    return {"build": build, "judge": judge, "score": lambda: score(a.tex)}[a.step]()


if __name__ == "__main__":
    raise SystemExit(main())
