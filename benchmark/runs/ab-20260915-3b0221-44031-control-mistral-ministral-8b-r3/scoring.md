# A/B scoring — ab-20260915-3b0221-44031-control-mistral-ministral-8b-r3

Dataset: openml-44031 (california), arm: control
Status: completed, script exit code: 0
Temperature: requested 1.0 (ab_protocol.md §5), status sent, in force 1.0

## holdout_score (ab_protocol.md section 3)

Split: 15480 train / 5160 holdout rows at fraction 0.25, seed 20260915, stratified False, grouped on duplicate rows True.
Holdout rows present verbatim in train: 0 (§9 A4 holds this at 0).

r2 = 0.815159, from `house_price_model.joblib`.

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
- frameworks seeded: ['sklearn']
- first split at source line: 30
- de-duplication had to happen before source line: 17
- transformer fit at source lines: none
- duplicate rows in train.csv: 0
- minority-class fraction in train.csv: None
- target is the last column of train.csv: True

Scored by a frozen function over the emitted source. No model scored this.
