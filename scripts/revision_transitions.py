"""Per-rule defect transitions across the control+revise turn.

The paper says self-revision "did not repair a single defect". The results CSV
carries only the NET pre-revision count, which cannot distinguish "repaired
nothing" from "repaired A and introduced B". This rescores both the preserved
pre-revision script and the post-revision script rule by rule with the CURRENT
scorer, so the claim is either earned or has to be softened.

The pinned dataset facts come from run.json's `defects` block, which is where
the harness recorded them, so a changed dataset cannot silently alter a
historical verdict.
"""

import collections
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(r"C:\Users\SABUNIS\Desktop\ml-copilot")
sys.path.insert(0, str(ROOT / "benchmark"))
from rescore_ab import _target_from_split  # noqa: E402
from run_ab import check_defects  # noqa: E402

rows = list(csv.DictReader((ROOT / "benchmark/ab_results.csv").open(encoding="utf-8")))
cr = [r for r in rows if r["experiment_id"] == "ab-20260915-d52979-v3"]
print("control+revise runs in the panel:", len(cr))

fixed = collections.Counter()
introduced = collections.Counter()
persisted = collections.Counter()
skipped = []
n_pairs = 0
repairs = []
net = collections.Counter()

for r in cr:
    d = ROOT / "benchmark" / r["artifacts_path"]
    pre, post, rj = d / "emitted_pre_revision.py", d / "emitted.py", d / "run.json"
    if not (pre.exists() and post.exists() and rj.exists()):
        skipped.append((r["run_id"], "file missing"))
        continue
    run = json.loads(rj.read_text(encoding="utf-8"))
    blk = run.get("defects") or {}
    split = run.get("split") or {}
    target = run.get("target") or split.get("target") or _target_from_split(run)
    need = ("input_duplicate_rows", "input_minority_fraction", "target_is_last_column")
    if not target or any(k not in blk for k in need):
        skipped.append((r["run_id"], f"target={target!r} facts={[k for k in need if k in blk]}"))
        continue
    facts = dict(
        target=target,
        duplicate_rows=blk["input_duplicate_rows"],
        minority_fraction=blk["input_minority_fraction"],
        target_is_last_column=blk["target_is_last_column"],
    )

    def flags(path, facts=facts):
        out = check_defects(path.read_text(encoding="utf-8", errors="replace"), **facts)
        f = out["flags"] if isinstance(out, dict) and "flags" in out else out
        return {k for k, v in f.items() if v} if isinstance(f, dict) else set(f)

    a, b = flags(pre), flags(post)
    n_pairs += 1
    for rule in a - b:
        fixed[rule] += 1
    for rule in b - a:
        introduced[rule] += 1
    for rule in a & b:
        persisted[rule] += 1
    net["worse" if len(b) > len(a) else "better" if len(b) < len(a) else "same"] += 1
    if a - b:
        repairs.append((r["run_id"], sorted(a - b), sorted(b - a)))

print(f"pairs rescored: {n_pairs}   skipped: {len(skipped)}")
for s in skipped[:6]:
    print("   skip:", s)

if n_pairs == 0:
    print("\nNOTHING WAS RESCORED — no verdict can be drawn.")
    raise SystemExit(1)

print("\nnet per run:", dict(net))
print("FIXED      (flagged before, clean after):", dict(fixed) or "NONE")
print("INTRODUCED (clean before, flagged after):", dict(introduced) or "NONE")
print("PERSISTED  (flagged on both sides):     ", dict(persisted) or "NONE")
print(f"\nruns in which at least one rule was repaired: {len(repairs)}")
for rid, f, i in repairs:
    print(f"  {rid}\n     fixed={f}  introduced={i}")

print(
    "\nVERDICT:",
    f"EARNED — zero rule-instances repaired across {n_pairs} pairs"
    if not fixed
    else f"TOO STRONG — {sum(fixed.values())} rule-instances repaired",
)
