# A/B scoring — ab-20260915-795b90-1480-control+revise-ollama-qwen2.5-7b-r1

Dataset: openml-1480 (ilpd), arm: control+revise
Status: script_failed, script exit code: 1
Temperature: requested 1.0 (ab_protocol.md §5), status sent, in force 1.0

## Self-revision round (ab_protocol.md §9 A8)

- Turns: 2 (first turn identical to the `control` arm's)
- Second turn carries no mlcompass content: it asks the model to review and fix its own script, naming no concern
- Pre-revision defect count: 1 of 6
- Post-revision defect count: 1 of 6 (this is the scored script)

- Flags the revision moved: none

## holdout_score (ab_protocol.md section 3)

Split: 439 train / 144 holdout rows at fraction 0.25, seed 0, stratified True, grouped on duplicate rows True.
Holdout rows present verbatim in train: 0 (§9 A4 holds this at 0).

Blank. script did not complete (script_failed); nothing to load

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
- frameworks imported: ['sklearn']
- frameworks seeded: ['sklearn']
- first split at source line: 18
- de-duplication had to happen before source line: 14
- transformer fit at source lines: none
- duplicate rows in train.csv: 12
- minority-class fraction in train.csv: 0.2870159453302961
- target is the last column of train.csv: True

Scored by a frozen function over the emitted source. No model scored this.
