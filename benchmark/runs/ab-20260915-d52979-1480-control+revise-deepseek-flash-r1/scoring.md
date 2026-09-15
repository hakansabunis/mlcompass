# A/B scoring — ab-20260915-d52979-1480-control+revise-deepseek-flash-r1

Dataset: openml-1480 (ilpd), arm: control+revise
Status: completed, script exit code: 0
Temperature: requested 1.0 (ab_protocol.md §5), status sent, in force 1.0

## Self-revision round (ab_protocol.md §9 A8)

- Turns: 2 (first turn identical to the `control` arm's)
- Second turn carries no mlcompass content: it asks the model to review and fix its own script, naming no concern
- Pre-revision defect count: 2 of 6
- Post-revision defect count: 2 of 6 (this is the scored script)

- Flags the revision moved: none

## holdout_score (ab_protocol.md section 3)

Split: 437 train / 146 holdout rows at fraction 0.25, seed 20260915, stratified True, grouped on duplicate rows True.
Holdout rows present verbatim in train: 0 (§9 A4 holds this at 0).

roc_auc = 0.686470, from `model.joblib`.

## process_defects (ab_protocol.md section 4)

defect_count: 2 of 6

| defect | present |
| --- | --- |
| `no_seed` | no |
| `leak_fit_before_split` | no |
| `leak_duplicate_rows` | yes |
| `wrong_metric_for_imbalance` | no |
| `no_validation` | yes |
| `target_in_features` | no |

Checklist evidence:
- frameworks imported: ['sklearn']
- frameworks seeded: ['sklearn']
- first split at source line: no split found
- de-duplication had to happen before source line: n/a
- transformer fit at source lines: none
- duplicate rows in train.csv: 10
- minority-class fraction in train.csv: 0.28604118993135014
- target is the last column of train.csv: True

Scored by a frozen function over the emitted source. No model scored this.
