"""The three RQ5 numbers the external review asked for and the paper lacks:
per-arm denominators and execution outcomes, per-arm cost, and whether
self-revision repaired any individual defect (not just the net count).
"""

import csv
import pathlib
import statistics

ROOT = pathlib.Path(r"C:\Users\SABUNIS\Desktop\ml-copilot")
rows = list(csv.DictReader((ROOT / "benchmark/ab_results.csv").open(encoding="utf-8")))

PANEL = {"ab-20260915-3b0221-v3", "ab-20260915-d52979-v3"}
panel = [r for r in rows if r["experiment_id"] in PANEL]
ARMS = ["control", "control+revise", "advise", "advise+audit"]
RULES = [
    "leak_duplicate_rows",
    "no_validation",
    "target_in_features",
    "no_seed",
    "leak_fit_before_split",
    "wrong_metric_for_imbalance",
]

print("=" * 78)
print("1. PER-ARM DENOMINATORS AND EXECUTION OUTCOMES")
print("=" * 78)
print(f"{'arm':<16}{'runs':>6}{'ran ok':>8}{'failed':>8}{'scored':>8}{'defects':>9}{'per run':>9}")
for a in ARMS:
    rs = [r for r in panel if r["arm"] == a]
    ran = sum(1 for r in rs if r["status"] == "completed")
    failed = sum(1 for r in rs if r["status"] == "script_failed")
    scored = sum(1 for r in rs if (r["holdout_score"] or "").strip())
    d = sum(int(r["defect_count"] or 0) for r in rs)
    print(f"{a:<16}{len(rs):>6}{ran:>8}{failed:>8}{scored:>8}{d:>9}{d / len(rs):>9.2f}")

tot = len(panel)
print(
    f"\ntotal runs {tot}; ran {sum(1 for r in panel if r['status'] == 'completed')}; "
    f"failed {sum(1 for r in panel if r['status'] == 'script_failed')}; "
    f"scored {sum(1 for r in panel if (r['holdout_score'] or '').strip())}"
)

# Artifact loadability: completed but no score = artifact could not be loaded.
print("\nartifact could not be loaded (ran, but no holdout score), per arm:")
for a in ARMS:
    rs = [r for r in panel if r["arm"] == a]
    n = sum(1 for r in rs if r["status"] == "completed" and not (r["holdout_score"] or "").strip())
    print(f"  {a:<16}{n}")

print()
print("=" * 78)
print("2. COST PER ARM")
print("=" * 78)
print(f"{'arm':<16}{'median s':>10}{'mean s':>9}{'in tok':>9}{'out tok':>9}{'total tok':>11}")
for a in ARMS:
    rs = [r for r in panel if r["arm"] == a]
    secs = [float(r["runtime_seconds"]) for r in rs if (r["runtime_seconds"] or "").strip()]
    it = [int(r["input_tokens"]) for r in rs if (r["input_tokens"] or "").strip()]
    ot = [int(r["output_tokens"]) for r in rs if (r["output_tokens"] or "").strip()]
    print(
        f"{a:<16}{statistics.median(secs):>10.1f}{statistics.mean(secs):>9.1f}"
        f"{statistics.mean(it):>9.0f}{statistics.mean(ot):>9.0f}"
        f"{statistics.mean(it) + statistics.mean(ot):>11.0f}"
    )

print()
print("=" * 78)
print("3. DID SELF-REVISION REPAIR ANY INDIVIDUAL DEFECT?")
print("=" * 78)
cr = [r for r in panel if r["arm"] == "control+revise"]
print(f"control+revise runs: {len(cr)}")
pre_field = [r for r in cr if (r["defect_count_pre_revision"] or "").strip()]
print(f"rows carrying defect_count_pre_revision: {len(pre_field)}")
if pre_field:
    worse = same = better = 0
    for r in pre_field:
        a, b = int(r["defect_count_pre_revision"]), int(r["defect_count"])
        if b > a:
            worse += 1
        elif b < a:
            better += 1
        else:
            same += 1
    print(f"  net worse {worse}, net same {same}, net better {better}")
print(
    "\nNOTE: the CSV carries only the NET pre-revision count, not per-rule "
    "pre-revision flags, so a fixed-A/introduced-B swap is invisible here.\n"
    "Per-rule transitions need rescoring emitted_pre_revision.py per rule."
)

# Per-rule totals for the record.
print("\nper-rule totals, current panel:")
print(f"{'rule':<30}" + "".join(f"{a:>16}" for a in ARMS))
for rule in RULES:
    line = f"{rule:<30}"
    for a in ARMS:
        line += f"{sum(int(r[rule] or 0) for r in panel if r['arm'] == a):>16}"
    print(line)
