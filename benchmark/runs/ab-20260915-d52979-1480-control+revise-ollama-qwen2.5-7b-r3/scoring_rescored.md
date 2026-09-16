# Rescored: ab-20260915-d52979-1480-control+revise-ollama-qwen2.5-7b-r3

Experiment id for these rows: `ab-20260915-d52979-v3`.

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
