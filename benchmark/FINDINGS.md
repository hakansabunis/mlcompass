# Benchmark findings — index

Two benchmarks, two reports, one set of shared machinery. Start here.

| Read this | For |
| --- | --- |
| [`REPORT_2026-09-15.md`](REPORT_2026-09-15.md) | **Detection benchmark** — does mlcompass find known defects in a dataset? Also carries the correction of four scoring defects found in review, and the A/B leak finding in §9. |
| [`REPORT_AB_2026-09-15.md`](REPORT_AB_2026-09-15.md) | **A/B benchmark** — does an LLM write better training code when given mlcompass's findings? 108 runs, four models. |
| [`protocol.md`](protocol.md) | Detection protocol (v1.0, written by Yusuf Ünlü) |
| [`ab_protocol.md`](ab_protocol.md) | A/B protocol (v1.3, amendments A1–A5 in §9) |
| [`ground_truth.json`](ground_truth.json) | The issue registry the detection benchmark scores against (v1.1) |

---

## The four findings that changed something

**1. mlcompass detected none of the three ground-truth issues on first run.**
Zero warnings of any kind on all three datasets. Both misses were row-level
properties that eight column-level checks could not see: 215 duplicate rows
in 748 on OpenML 1464, and 965 of 20640 rows sitting at the target's exact
maximum on 44031. Two detectors were added and recall went 0/3 → 3/3. The
same detectors then fired on `scripts/data/wine_quality_red.csv` — 240
duplicates in 1599 rows — which had been in the repository the whole time.

**2. Recall was not measuring the LLM.** Scoring read the whole terminal
output, but the deterministic layer prints its warnings before the LLM runs.
In one run `duplicate row` appears at character 1777 and the advisor's
failure at 2078 — it scored 1/1 on text the LLM never produced. Every LLM
lane was inheriting the deterministic layer's detections, which is why they
all read 9/9. Scored on what each model actually wrote, they separate from
1/9 to 9/9.

**3. The A/B benchmark was rewarding the leak it was built to penalise.**
A plain split put 71 of 187 holdout rows verbatim into train, so a script
that correctly de-duplicated scored *below* one that did not — 0.5552
against 0.6532–0.6835, across twelve runs, with nothing in between. Fixed
by splitting on duplicate groups: 0 holdout rows in train, 125 duplicates
still inside train, so the score is honest and the defect stays reachable.

**4. The A/B result, on 108 runs.** The defect count falls from `control`
to `advise+audit` on all four model lanes. The held-out score shows no
direction — 3 cells better, 3 worse, 3 flat — and the spread within an arm
repeatedly exceeds the gap between arms. This is the outcome
`ab_protocol.md` §6 named before any data existed: **the process improves,
the metric does not.**

---

## How to read a number here

Every rate traces to a row in `results.csv` or `ab_results.csv`, and every
row names an evidence directory under `runs/` holding the command, the
prompt, the model's verbatim reply, the emitted script, its stdout and exit
code, and the scoring. Nothing in either report was typed by hand from a
recollection.

Three things are deliberately **not** scored, and each says why in its own
report: `false_positives` and `hallucinations` in the detection benchmark
(refuting a claim needs a reviewer), and the held-out score for
`gpt-5.4-mini` in the A/B benchmark (a scorer limitation, open as amendment
A6, left unfixed because changing a measure after seeing results is what the
frozen plan exists to prevent).

## Open

- **A6** — whether a saved bundle (model + encoder + columns) counts as a
  scoreable artefact. 15 runs scored blank for arguably doing the more
  correct thing.
- **A decision rule for "beats."** `ab_protocol.md` §9 records that there
  isn't one. With n=3 and medians separated by 0.0002, this is the binding
  constraint on reporting, not the amount of data.
- **Ground truth for the remaining datasets.** 22 are selected in
  `datasets.md`; 3 have verified issues. That is the real cost of this
  benchmark, and it is manual work rather than compute.
