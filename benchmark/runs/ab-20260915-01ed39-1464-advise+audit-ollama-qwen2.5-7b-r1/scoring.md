# A/B scoring — ab-20260915-01ed39-1464-advise+audit-ollama-qwen2.5-7b-r1

Dataset: openml-1464 (blood-transfusion-service-center), arm: advise+audit
Status: completed, script exit code: 0
Temperature: requested 1.0 (ab_protocol.md §5), status sent, in force 1.0

## Revision round (ab_protocol.md section 2, §9 A1)

- Turns: 2 (first turn identical to the `advise` arm's)
- `mlcompass audit` on this arm's own first-turn script: ok
- Pre-revision defect count: 0 of 6
- Post-revision defect count: 0 of 6 (this is the scored script)

- Flags the revision moved: none

## holdout_score (ab_protocol.md section 3)

roc_auc = 0.555229, from `trained_model.joblib`.

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
- first split at source line: 21
- de-duplication had to happen before source line: 17
- transformer fit at source lines: none
- duplicate rows in train.csv: 142
- minority-class fraction in train.csv: 0.23885918003565063
- target is the last column of train.csv: True

Scored by a frozen function over the emitted source. No model scored this.
