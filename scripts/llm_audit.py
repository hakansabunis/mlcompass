"""Analysis plans A12 and A13: the blind semantic audit by LLM judges.

Each judge receives, for each of the 204 items of the A7 sheet, what the human
sheet shows (in English): the instructions with labels A-E and their decision
rules, E's table, E's other names, the frame's columns, the whole response and
the marked element. No arm, model, provider, date or stratum. A12 asked two
judges (gpt, deepseek); A13 asks one stronger judge (gpt55) the same question
with the same input.

    python -X utf8 scripts/llm_audit.py judge [--only gpt55]   # appends to llm_labels.jsonl, resumes
    python -X utf8 scripts/llm_audit.py score [--tex paper/tse_latex/table_llm_rows.tex]
    python -X utf8 scripts/llm_audit.py score --judges gpt55 --out llm_results_a13.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import time
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDIT = ROOT / "benchmark" / "semantic_audit_A7"
JUDGES = {
    "gpt": {"model": "gpt-5.4-mini", "key_env": "OPENAI_API_KEY", "base_url": None, "temperature": None},
    # A12a: Mistral answered 429 to every call, so the second judge is DeepSeek.
    "deepseek": {"model": "deepseek-chat", "key_env": "DEEPSEEK_API_KEY",
                 "base_url": "https://api.deepseek.com", "temperature": 0.0},
    # A13: one stronger judge, the same instrument; stops at its registered spend cap.
    "gpt55": {"model": "gpt-5.5-2026-04-23", "key_env": "OPENAI_API_KEY", "base_url": None,
              "temperature": None, "reasoning_effort": "medium", "max_completion_tokens": 6000,
              "usd_per_m": (5.0, 30.0), "cap_usd": 15.0},
}
A12_JUDGES = ("gpt", "deepseek")
NOT_FALSE = {"S4 real column E does not list", "S8 control: admissible name",
             "S9 control: claim that matches E"}
FALSE_OR_ABSENT = {"S3 wrong number, another item's value", "S5 invented name, list written in advance",
                   "S6 invented name", "S7 number E records nowhere"}

INSTRUCTIONS = """You are one of two independent annotators in a blind audit. Each item is one element of an answer that a language model gave about some evidence E: either a name the answer cites, or a claim (column, statistic, value). You see the evidence E, the whole answer for context, and the marked element. You do not see which system produced the answer.

Question: what does the marked element say about the evidence? Choose exactly one label.
A - Supported, in the right place: E states this, in this place (a column that E lists, or a claim that reports E's value within 0.005).
B - Right information, wrong place: what it says is true of E (or of the data), but it sits in a field or under a key that does not fit, and a reader would not be misled.
C - False or unsupported relation: built from things E contains, but the relation it states is not in E or contradicts E; a reader could be misled.
D - A name or number that is not in the evidence: a name that appears neither in E nor in the frame's column list, or a number that appears nowhere in E.
E - Cannot tell. Choose this if you are undecided.

Rules: judge only the marked element; the rest of the answer is context. Never choose D for a name that appears in E or in the frame's column list. Choose B only if a reader would not be misled.

An example unrelated to the data: let the evidence be a weather report, "Ankara: high 31, low 18; Izmir: high 35, low 22." In an answer, "Ankara, high, 31" is A. Writing "high" in the "cities" field is B if what is meant is clear. "Izmir, low, 18" (Ankara's value given to Izmir) is C. "Bursa, high, 30" or "Ankara, high, 29" is D. Decide for your own item which label fits; the example only shows what the labels mean.

Reply with JSON only: {"label": "A|B|C|D|E", "reason": "one short sentence"}."""


def load_sheet() -> list[dict]:
    html = (AUDIT / "audit_sheet.html").read_text(encoding="utf-8")
    return json.loads(re.search(r"const ITEMS = (\[.*?\]);\nconst CHOICES", html, flags=re.S).group(1))


def render(it: dict) -> str:
    ev = it["evidence"]
    lines = ["EVIDENCE E: columns, statistics and values (column | statistic | value):"]
    lines += [f"  {r[0]} | {r[1]} | {r[2]}" for r in ev["values"]]
    lines.append("Other names in E (not columns): " + (", ".join(ev["other_names"]) or "none"))
    lines.append("Columns of the data frame: " + (", ".join(ev["frame"]) or "none"))
    lines.append("")
    lines.append("THE WHOLE ANSWER (context):")
    lines.append(f"  verdict: {it['verdict']}")
    lines.append("  cited columns: " + (", ".join(it["columns_referenced"]) or "none"))
    for i, c in enumerate(it["claims"]):
        lines.append(f"  claim {i}: column={c.get('column')}, statistic={c.get('statistic')}, value={c.get('value')}")
    lines.append("")
    if it["target_type"] == "name":
        lines.append(f"MARKED ELEMENT: the cited name \"{it['target']}\" "
                     "(in the answer's cited-column list or in a claim's column field).")
    else:
        c = it["claims"][int(it["target"])]
        lines.append(f"MARKED ELEMENT: claim {it['target']}: column={c.get('column')}, "
                     f"statistic={c.get('statistic')}, value={c.get('value')}.")
    return "\n".join(lines)


def ask(client, cfg: dict, item_text: str) -> dict:
    kwargs = {"model": cfg["model"],
              "messages": [{"role": "system", "content": INSTRUCTIONS},
                           {"role": "user", "content": item_text}],
              "response_format": {"type": "json_object"}}
    if cfg["temperature"] is not None:
        kwargs["temperature"] = cfg["temperature"]
    for opt in ("reasoning_effort", "max_completion_tokens"):
        if cfg.get(opt) is not None:
            kwargs[opt] = cfg[opt]
    for attempt in range(4):
        try:
            r = client.chat.completions.create(**kwargs)
            text = r.choices[0].message.content or ""
            out = json.loads(text[text.find("{"):text.rfind("}") + 1])
            label = str(out.get("label", "")).strip().upper()[:1]
            if label not in "ABCDE" or not label:
                raise ValueError(f"bad label {out!r}")
            usage = [r.usage.prompt_tokens, r.usage.completion_tokens] if r.usage else None
            details = getattr(r.usage, "completion_tokens_details", None) if r.usage else None
            if usage and getattr(details, "reasoning_tokens", None) is not None:
                usage.append(details.reasoning_tokens)
            return {"label": label, "reason": str(out.get("reason", ""))[:300], "usage": usage,
                    "served_model": getattr(r, "model", None)}
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                return {"label": "ERROR", "reason": f"{type(e).__name__}: {str(e)[:200]}", "usage": None}
            time.sleep(5 * (attempt + 1))
    return {"label": "ERROR", "reason": "unreachable", "usage": None}


def judge(only: str | None) -> int:
    from openai import OpenAI  # noqa: PLC0415

    items = load_sheet()
    log = AUDIT / "llm_labels.jsonl"
    done = set()
    spent: dict[str, float] = defaultdict(float)
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec["label"] != "ERROR":
                    done.add((rec["judge"], str(rec["id"])))
                spent[rec["judge"]] += cost(rec["judge"], rec.get("usage"))
    for short, cfg in JUDGES.items():
        if only and short != only:
            continue
        key = os.environ.get(cfg["key_env"])
        if not key:
            raise SystemExit(f"{cfg['key_env']} is not set (source scripts/load_keys.sh)")
        client = OpenAI(api_key=key, base_url=cfg["base_url"]) if cfg["base_url"] else OpenAI(api_key=key)
        todo = [it for it in items if (short, str(it["id"])) not in done]
        for k, it in enumerate(todo, 1):
            if cfg.get("cap_usd") is not None and spent[short] >= cfg["cap_usd"]:
                print(f"{short}: spend cap {cfg['cap_usd']} USD reached after {k - 1} items; stopping",
                      flush=True)
                return 0
            res = ask(client, cfg, render(it))
            spent[short] += cost(short, res.get("usage"))
            with log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"judge": short, "model": cfg["model"], "id": it["id"], **res}) + "\n")
            if k % 20 == 0:
                print(f"{short}: {k}/{len(todo)}, {spent[short]:.2f} USD", flush=True)
        print(f"{short}: done ({len(todo)} new), {spent[short]:.2f} USD", flush=True)
    return 0


def cost(judge_name: str, usage: list | None) -> float:
    """Spend in USD of one call at the judge's listed price (0 where none is listed)."""
    price = JUDGES.get(judge_name, {}).get("usd_per_m")
    if not usage or not price:
        return 0.0
    return (usage[0] * price[0] + usage[1] * price[1]) / 1e6


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


def score(tex: str | None, judges: tuple[str, ...] = A12_JUDGES, out: str = "llm_results.json") -> int:
    key = json.loads((AUDIT / "KEY_sealed.json").read_text(encoding="utf-8"))
    labels: dict[str, dict[str, str]] = defaultdict(dict)
    for line in (AUDIT / "llm_labels.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec["label"] != "ERROR" and rec["judge"] in judges:
                labels[str(rec["id"])][rec["judge"]] = rec["label"]
    strata = sorted({k["stratum"] for k in key.values()})
    res: dict = {"by_stratum": {}, "validity": {}, "missing": {}}
    for j in judges:
        by = defaultdict(Counter)
        for i, k in key.items():
            by[k["stratum"]][labels[i].get(j, "missing")] += 1
        res["by_stratum"][j] = {s: dict(by[s]) for s in strata}
        nf_n = sum(sum(by[s].values()) for s in NOT_FALSE)
        nf_ok = sum(by[s]["A"] + by[s]["B"] for s in NOT_FALSE)
        fa_n = sum(sum(by[s].values()) for s in FALSE_OR_ABSENT)
        fa_ok = sum(by[s]["C"] + by[s]["D"] for s in FALSE_OR_ABSENT)
        res["validity"][j] = {"not_false_AB": [nf_ok, nf_n, wilson(nf_ok, nf_n)],
                              "false_or_absent_CD": [fa_ok, fa_n, wilson(fa_ok, fa_n)],
                              "valid": nf_ok >= 0.9 * nf_n and fa_ok >= 0.9 * fa_n}
        res["missing"][j] = sum(1 for i in key if j not in labels[i])
    if len(judges) == 2:
        both = [i for i in key if all(j in labels[i] for j in judges)]
        j1, j2 = judges
        res["kappa"] = kappa([labels[i][j1] for i in both], [labels[i][j2] for i in both])
        res["agreement"] = sum(labels[i][j1] == labels[i][j2] for i in both) / len(both)
        cons = defaultdict(Counter)
        for i in both:
            if labels[i][j1] == labels[i][j2]:
                cons[key[i]["stratum"]][labels[i][j1]] += 1
            else:
                cons[key[i]["stratum"]]["disagree"] += 1
        res["consensus"] = {s: dict(cons[s]) for s in strata}
    else:
        res["shares"] = {j: {s: {x: [c.get(x, 0), sum(c.values()), wilson(c.get(x, 0), sum(c.values()))]
                                 for x in "ABCDE"}
                             for s, c in res["by_stratum"][j].items()} for j in judges}
    preds = {}
    for j in judges:
        b = res["by_stratum"][j]
        s1, s3 = b.get("S1 misfiled name", {}), b.get("S3 wrong number, another item's value", {})
        ctrl = Counter(b.get("S8 control: admissible name", {})) + Counter(b.get("S9 control: claim that matches E", {}))
        preds[j] = {"P1_S1_majority_B": s1.get("B", 0) > sum(s1.values()) / 2,
                    "P2_S3_majority_C": s3.get("C", 0) > sum(s3.values()) / 2,
                    "P3_controls_A_90": ctrl.get("A", 0) >= 0.9 * sum(ctrl.values())}
    res["predictions"] = preds
    (AUDIT / out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    if tex:
        lines = []
        for s in strata:
            n = sum(res["by_stratum"][judges[0]][s].values())
            cells = []
            for j in judges:
                c = res["by_stratum"][j][s]
                cells += [str(c.get(x, 0)) for x in "ABCDE"]
            name = s.split(" ", 1)[1].replace("E", "$E$")
            lines.append(f"{s.split(' ', 1)[0]} {name} & {n} & " + " & ".join(cells) + r" \\")
        pathlib.Path(tex).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {tex}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["judge", "score"])
    ap.add_argument("--only", choices=sorted(JUDGES), default=None)
    ap.add_argument("--tex", default=None)
    ap.add_argument("--judges", nargs="+", choices=sorted(JUDGES), default=list(A12_JUDGES))
    ap.add_argument("--out", default="llm_results.json")
    a = ap.parse_args()
    return judge(a.only) if a.step == "judge" else score(a.tex, tuple(a.judges), a.out)


if __name__ == "__main__":
    raise SystemExit(main())
