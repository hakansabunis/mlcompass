# A/B scoring — ab-20260915-3b0221-1464-advise-ollama-qwen2.5-7b-r1

Dataset: openml-1464 (blood-transfusion-service-center), arm: advise
Status: completed, script exit code: 0
Temperature: requested 1.0 (ab_protocol.md §5), status sent, in force 1.0

## holdout_score (ab_protocol.md section 3)

Split: 524 train / 224 holdout rows at fraction 0.25, seed 20260915, stratified True, grouped on duplicate rows True.
Holdout rows present verbatim in train: 0 (§9 A4 holds this at 0).

Blank. no saved artefact could be loaded and used to predict on the holdout
- tried `trained_model.joblib`: ValueError: model classes [np.int64(0), np.int64(1)] do not match the holdout labels [1, 2]

## process_defects (ab_protocol.md section 4)

defect_count: 2 of 6

| defect | present |
| --- | --- |
| `no_seed` | no |
| `leak_fit_before_split` | yes |
| `leak_duplicate_rows` | yes |
| `wrong_metric_for_imbalance` | no |
| `no_validation` | no |
| `target_in_features` | no |

Checklist evidence:
- frameworks imported: ['sklearn']
- frameworks seeded: ['sklearn']
- first split at source line: 20
- de-duplication had to happen before source line: 16
- transformer fit at source lines: [13]
- duplicate rows in train.csv: 125
- minority-class fraction in train.csv: 0.25
- target is the last column of train.csv: True

Scored by a frozen function over the emitted source. No model scored this.
