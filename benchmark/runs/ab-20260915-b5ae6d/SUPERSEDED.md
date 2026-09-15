# SUPERSEDED — the split under `_splits/` is the contaminated one

This experiment was planned and run under A/B 1.1, before
`ab_protocol.md` §9 A4.

`_splits/openml-1464-seed20260915/holdout.csv` shares **71 of its 187 rows
verbatim** with the `train.csv` beside it, because the split was drawn per
row rather than per duplicate group. Every run under this experiment id was
scored against that file, so none of their `holdout_score` values is a
result. `plan.md` describes the geometry that produced it and is accurate
about what was done — it is the geometry itself that was wrong.

Kept rather than deleted per §9 A4: discarding the evidence of a design error
is how the error survives. Superseded by A/B 1.2, which splits on duplicate
groups and records the contamination count in every run record.
