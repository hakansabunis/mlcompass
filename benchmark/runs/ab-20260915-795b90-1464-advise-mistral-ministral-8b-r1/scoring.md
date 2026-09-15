# A/B scoring — ab-20260915-795b90-1464-advise-mistral-ministral-8b-r1

Dataset: openml-1464 (blood-transfusion-service-center), arm: advise
Status: completed, script exit code: 0
Temperature: requested 1.0 (ab_protocol.md §5), status sent, in force 1.0

## holdout_score (ab_protocol.md section 3)

Split: 588 train / 160 holdout rows at fraction 0.25, seed 0, stratified True, grouped on duplicate rows True.
Holdout rows present verbatim in train: 0 (§9 A4 holds this at 0).

roc_auc = 0.665312, from `random_forest_classifier.joblib`.

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
- frameworks imported: ['sklearn']
- frameworks seeded: ['sklearn']
- first split at source line: 23
- de-duplication had to happen before source line: 19
- transformer fit at source lines: none
- duplicate rows in train.csv: 189
- minority-class fraction in train.csv: 0.23469387755102042
- target is the last column of train.csv: True

Scored by a frozen function over the emitted source. No model scored this.
