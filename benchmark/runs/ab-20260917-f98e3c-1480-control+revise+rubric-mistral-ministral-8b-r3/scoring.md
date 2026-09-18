# A/B scoring — ab-20260917-f98e3c-1480-control+revise+rubric-mistral-ministral-8b-r3

Dataset: openml-1480 (ilpd), arm: control+revise+rubric
Status: script_failed, script exit code: 1
Temperature: requested 1.0 (ab_protocol.md §5), status sent, in force 1.0

## Rubric revision round (ab_protocol.md §9 A17)

- Turns: 2 (first turn identical to the `control` arm's)
- Second turn carries no mlcompass content: the six scored concerns are named as a generic rubric, with no claim about which of them this script has
- Pre-revision defect count: 1 of 6
- Post-revision defect count: 0 of 6 (this is the scored script)

- Flags the revision moved: `leak_duplicate_rows` fixed

## holdout_score (ab_protocol.md section 3)

Split: 436 train / 147 holdout rows at fraction 0.25, seed 2, stratified True, grouped on duplicate rows True.
Holdout rows present verbatim in train: 0 (§9 A4 holds this at 0).

Blank. script did not complete (script_failed); nothing to load

## process_defects (ab_protocol.md section 4)

defect_count: 0 of 6

| defect | present |
| --- | --- |
| `no_seed` | no |
| `leak_fit_before_split` | no |
| `leak_duplicate_rows` | no |
| `wrong_metric_for_imbalance` | no |
| `no_validation` | no |
| `target_in_features` | no |

Checklist evidence:
- frameworks imported: ['numpy', 'sklearn']
- frameworks seeded: ['numpy', 'sklearn']
- first split at source line: 52
- de-duplication had to happen before source line: 26
- transformer fit at source lines: none
- duplicate rows in train.csv: 9
- minority-class fraction in train.csv: 0.286697247706422
- target is the last column of train.csv: True

Scored by a frozen function over the emitted source. No model scored this.
