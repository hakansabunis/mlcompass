# A/B scoring — ab-20260915-3b0221-1480-advise+audit-ollama-qwen2.5-7b-r2

Dataset: openml-1480 (ilpd), arm: advise+audit
Status: script_failed, script exit code: 1
Temperature: requested 1.0 (ab_protocol.md §5), status sent, in force 1.0

## Revision round (ab_protocol.md section 2, §9 A1)

- Turns: 2 (first turn identical to the `advise` arm's)
- `mlcompass audit` on this arm's own first-turn script: ok
- Pre-revision defect count: 0 of 6
- Post-revision defect count: 0 of 6 (this is the scored script)

- Flags the revision moved: none

## holdout_score (ab_protocol.md section 3)

Split: 437 train / 146 holdout rows at fraction 0.25, seed 20260915, stratified True, grouped on duplicate rows True.
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
- frameworks imported: ['sklearn']
- frameworks seeded: ['sklearn']
- first split at source line: 18
- de-duplication had to happen before source line: 14
- transformer fit at source lines: none
- duplicate rows in train.csv: 10
- minority-class fraction in train.csv: 0.28604118993135014
- target is the last column of train.csv: True

Scored by a frozen function over the emitted source. No model scored this.
