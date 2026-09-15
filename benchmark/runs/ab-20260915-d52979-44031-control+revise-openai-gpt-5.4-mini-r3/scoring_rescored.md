# Rescored: ab-20260915-d52979-44031-control+revise-openai-gpt-5.4-mini-r3

Experiment id for these rows: `ab-20260915-d52979-v2`.

Re-scored from the preserved `emitted.py` with the corrected
`target_in_features` rule (amendment A7). Nothing was re-executed and
no model was called; the original `scoring.md` is kept beside this file.

Defect count: **1 -> 0**

| flag | before | after |
| --- | :---: | :---: |
| `no_seed` | yes | no **<-- changed** |
| `leak_fit_before_split` | no | no |
| `leak_duplicate_rows` | no | no |
| `wrong_metric_for_imbalance` | no | no |
| `no_validation` | no | no |
| `target_in_features` | no | no |

Changed: `no_seed`.
