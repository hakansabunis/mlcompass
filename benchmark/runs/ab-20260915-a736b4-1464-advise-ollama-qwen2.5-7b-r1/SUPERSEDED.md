# SUPERSEDED — this run is not a result

Scored under A/B 1.1, before `ab_protocol.md` §9 A4.

The holdout it was scored on was drawn **per row**, not per duplicate group.
On OpenML 1464 that puts **71 of 187 holdout rows verbatim into train**, so
the score below is partly a measurement of memorisation:

- a script that de-duplicated was scored on a holdout whose answers it had
  thrown away, and
- a script that left the duplicates in was scored on rows it had kept.

Across the twelve 1.1 runs the two groups separate completely — every one of
the eight leaking scripts scored at least 0.6532, every one of the four clean
scripts scored 0.5552, nothing in between. `holdout_score` is the measure §1
calls the one that cannot be argued with, and here it was inverted.

**Do not read `holdout_score` in `run.json`, `scoring.md` or
`ab_results_v1.1_superseded.csv` as a result.** The defect flags are
unaffected — §4's checklist is a function of the emitted source, not of the
split — and `emitted.py` is still the artefact §7 preserves.

Kept rather than deleted per §9 A4. Superseded by the runs under A/B 1.2,
whose splits carry `holdout_rows_present_in_train: 0`.

This run: arm `advise`, qwen2.5:7b via ollama-qwen2.5-7b, 561 train / 187 holdout rows.
