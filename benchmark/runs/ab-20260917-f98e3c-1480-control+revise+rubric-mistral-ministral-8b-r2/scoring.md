# A/B scoring — ab-20260917-f98e3c-1480-control+revise+rubric-mistral-ministral-8b-r2

Dataset: openml-1480 (ilpd), arm: control+revise+rubric
Status: completed, script exit code: 0
Temperature: requested 1.0 (ab_protocol.md §5), status sent, in force 1.0

## Rubric revision round (ab_protocol.md §9 A17)

- Turns: 2 (first turn identical to the `control` arm's)
- Second turn carries no mlcompass content: the six scored concerns are named as a generic rubric, with no claim about which of them this script has
- Pre-revision defect count: 1 of 6
- Post-revision defect count: 0 of 6 (this is the scored script)

- Flags the revision moved: `leak_duplicate_rows` fixed

## holdout_score (ab_protocol.md section 3)

Split: 437 train / 146 holdout rows at fraction 0.25, seed 1, stratified True, grouped on duplicate rows True.
Holdout rows present verbatim in train: 0 (§9 A4 holds this at 0).

roc_auc = 0.652381, from `trained_classifier.joblib`.

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
- first split at source line: 32
- de-duplication had to happen before source line: 24
- transformer fit at source lines: none
- duplicate rows in train.csv: 10
- minority-class fraction in train.csv: 0.28832951945080093
- target is the last column of train.csv: True

Scored by a frozen function over the emitted source. No model scored this.
