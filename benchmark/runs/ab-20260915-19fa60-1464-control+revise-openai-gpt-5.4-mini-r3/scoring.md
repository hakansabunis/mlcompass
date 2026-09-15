# A/B scoring — ab-20260915-19fa60-1464-control+revise-openai-gpt-5.4-mini-r3

Dataset: openml-1464 (blood-transfusion-service-center), arm: control+revise
Status: completed, script exit code: 0
Temperature: requested 1.0 (ab_protocol.md §5), status sent, in force 1.0

## Self-revision round (ab_protocol.md §9 A8)

- Turns: 2 (first turn identical to the `control` arm's)
- Second turn carries no mlcompass content: it asks the model to review and fix its own script, naming no concern
- Pre-revision defect count: 1 of 6
- Post-revision defect count: 1 of 6 (this is the scored script)

- Flags the revision moved: none

## holdout_score (ab_protocol.md section 3)

Split: 559 train / 189 holdout rows at fraction 0.25, seed 2, stratified True, grouped on duplicate rows True.
Holdout rows present verbatim in train: 0 (§9 A4 holds this at 0).

Blank. no saved artefact could be loaded and used to predict on the holdout
- tried `trained_model.joblib`: loaded object has no .predict (dict)

## process_defects (ab_protocol.md section 4)

defect_count: 1 of 6

| defect | present |
| --- | --- |
| `no_seed` | no |
| `leak_fit_before_split` | no |
| `leak_duplicate_rows` | yes |
| `wrong_metric_for_imbalance` | no |
| `no_validation` | no |
| `target_in_features` | no |

Checklist evidence:
- frameworks imported: ['numpy', 'sklearn']
- frameworks seeded: ['sklearn']
- first split at source line: 29
- de-duplication had to happen before source line: 24
- transformer fit at source lines: none
- duplicate rows in train.csv: 160
- minority-class fraction in train.csv: 0.24508050089445438
- target is the last column of train.csv: True

Scored by a frozen function over the emitted source. No model scored this.
