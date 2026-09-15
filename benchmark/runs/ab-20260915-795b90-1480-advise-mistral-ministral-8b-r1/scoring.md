# A/B scoring — ab-20260915-795b90-1480-advise-mistral-ministral-8b-r1

Dataset: openml-1480 (ilpd), arm: advise
Status: completed, script exit code: 0
Temperature: requested 1.0 (ab_protocol.md §5), status sent, in force 1.0

## holdout_score (ab_protocol.md section 3)

Split: 439 train / 144 holdout rows at fraction 0.25, seed 0, stratified True, grouped on duplicate rows True.
Holdout rows present verbatim in train: 0 (§9 A4 holds this at 0).

roc_auc = 0.780369, from `trained_classifier.joblib`.

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
- first split at source line: 54
- de-duplication had to happen before source line: 23
- transformer fit at source lines: none
- duplicate rows in train.csv: 12
- minority-class fraction in train.csv: 0.2870159453302961
- target is the last column of train.csv: True

Scored by a frozen function over the emitted source. No model scored this.
