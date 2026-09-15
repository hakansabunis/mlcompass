# Rescored: ab-20260915-3b0221-1480-control-mistral-ministral-8b-r1

Experiment id for these rows: `ab-20260915-3b0221-rescored-v2`.

Re-scored from the preserved `emitted.py` with the corrected
`target_in_features` rule (amendment A7). Nothing was re-executed and
no model was called; the original `scoring.md` is kept beside this file.

Defect count: **1 -> 1**

| flag | before | after |
| --- | :---: | :---: |
| `no_seed` | no | no |
| `leak_fit_before_split` | no | no |
| `leak_duplicate_rows` | yes | yes |
| `wrong_metric_for_imbalance` | no | no |
| `no_validation` | no | no |
| `target_in_features` | no | no |

No flag changed.
