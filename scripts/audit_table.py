"""LaTeX rows of the semantic-audit table: the labels of the judge that passed
its validity check (A13) per stratum, with the strata corrected for fault 17,
and every judge's validity check on the items whose truth is fixed.

    python -X utf8 scripts/audit_table.py [--tex paper/tse_latex/table_audit_rows.tex]

Reads benchmark/semantic_audit_A7/{llm_results_a13,llm_results,nli_results}.json,
which scripts/llm_audit.py and scripts/nli_audit.py write.
"""

from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDIT = ROOT / "benchmark" / "semantic_audit_A7"
SHORT = {
    "S1 misfiled name": "S1 misfiled name",
    "S2 number E records under another pair": "S2 number under another pair",
    "S3 wrong number, another item's value": "S3 another item's number",
    "S4 real column E does not list": "S4 real column $E$ lacks",
    "S5 invented name, list written in advance": "S5 invented, stale list",
    "S6 invented name": "S6 invented name",
    "S7 number E records nowhere": "S7 number in no part of $E$",
    "S8 control: admissible name": "S8 admissible name",
    "S9 control: claim that matches E": "S9 claim that matches $E$",
}
JUDGES = [("nli_results.json", "deberta", "DeBERTa (NLI)"), ("nli_results.json", "roberta", "RoBERTa (NLI)"),
          ("llm_results.json", "gpt", r"\texttt{gpt-5.4-mini}"),
          ("llm_results.json", "deepseek", r"\texttt{deepseek-chat}"),
          ("llm_results_a13.json", "gpt55", r"\texttt{gpt-5.5}")]


def rows() -> list[str]:
    a13 = json.loads((AUDIT / "llm_results_a13.json").read_text(encoding="utf-8"))
    out = []
    for stratum, counts in a13["by_stratum_corrected"]["gpt55"].items():
        n = sum(counts.values())
        out.append(f"{SHORT[stratum]} & {n} & " + " & ".join(str(counts.get(x, 0)) for x in "ABCDE") + r" \\")
    out.append(r"\midrule")
    out.append(r"Judge & \multicolumn{2}{r}{kept} & \multicolumn{2}{r}{caught} & \multicolumn{2}{r}{} \\")
    out.append(r"\midrule")
    for fname, j, label in JUDGES:
        v = json.loads((AUDIT / fname).read_text(encoding="utf-8"))["validity_corrected"][j]
        kept = v.get("known_supported") or v.get("not_false_AB")
        caught = v.get("known_unsupported") or v.get("false_or_absent_CD")
        mark = r"\textbf{valid}" if v["valid"] else "invalid"
        out.append(f"{label} & \\multicolumn{{2}}{{r}}{{{kept[0]}/{kept[1]}}} & "
                   f"\\multicolumn{{2}}{{r}}{{{caught[0]}/{caught[1]}}} & "
                   f"\\multicolumn{{2}}{{r}}{{{mark}}} \\\\")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=None)
    a = ap.parse_args()
    text = "\n".join(rows()) + "\n"
    print(text)
    if a.tex:
        pathlib.Path(a.tex).write_text(text, encoding="utf-8")
        print(f"wrote {a.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
