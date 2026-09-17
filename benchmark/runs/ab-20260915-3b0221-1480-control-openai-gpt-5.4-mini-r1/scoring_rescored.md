# Rescored: ab-20260915-3b0221-1480-control-openai-gpt-5.4-mini-r1

Experiment id for these rows: `v4`.

Re-scored from the preserved `emitted.py` with the corrected
`target_in_features` rule (amendment A7). Nothing was re-executed and
no model was called; the original `scoring.md` is kept beside this file.

Defect count: **2 -> 1**

| flag | before | after |
| --- | :---: | :---: |
| `no_seed` | yes | no **<-- changed** |
| `leak_fit_before_split` | no | no |
| `leak_duplicate_rows` | yes | yes |
| `wrong_metric_for_imbalance` | no | no |
| `no_validation` | no | no |
| `target_in_features` | no | no |

Changed: `no_seed`.
