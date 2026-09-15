# MLCompass benchmark

This directory provides templates for future academic evaluation of MLCompass
on public datasets with documented ML/data problems. It contains no experimental
results, dataset downloads, or automated benchmark runner. Existing MLCompass
functionality is unchanged.

## Contents

- `datasets.md`: dataset metadata, evidence for known issues, and expected findings.
- `protocol.md`: common execution, scoring, and reporting procedure.
- `results.csv`: one row per future dataset case, configuration, and repetition.
- `runs/`: supporting records for each run, stored under a unique run ID.

A second, separate battery asks a different question — whether an LLM writes
better training code with mlcompass's findings in front of it — and keeps its
own protocol, harness and result file:

- `ab_protocol.md`: the three-arm with/without protocol, currently A/B 1.2.
- `run_ab.py`: its harness. Reads the frozen constants back out of the
  protocol before every run and refuses to start if the two disagree.
- `ab_results.csv`: one row per (dataset x arm x model x repetition).

## Superseded evidence

Some preserved runs are **not results** and are marked as such rather than
deleted, per `ab_protocol.md` §9 A4: discarding the evidence of a design
error is how the error survives.

- `ab_results_v1.0.csv` and `ab_results_v1.1_superseded.csv` hold every A/B
  run made before A/B 1.2.
- Every `runs/ab-*` directory from those versions carries a `SUPERSEDED.md`
  saying what is wrong with it.

What is wrong with them is one thing: the holdout was drawn per row rather
than per duplicate group, so 71 of its 187 rows on OpenML 1464 appear
verbatim in train. Their `holdout_score` values measure memorisation as much
as skill and must not be read as results. The emitted scripts and the
defect-checklist flags in those runs are unaffected — §4's checklist is a
function of the source, not of the split.

The local `.gitignore` overrides the repository's generic `runs/` exclusion;
`runs/.gitkeep` preserves the initially empty directory.

## Getting started

1. Copy the dataset entry template and document a verified evaluation case.
2. Freeze the experiment settings and scoring rules described in `protocol.md`.
3. Execute each planned configuration and repetition in a fresh workspace.
4. Save evidence under `runs/<run_id>/`, review findings, and append a results row.

Keep source data at its documented location unless redistribution is permitted.
Do not commit credentials or private data. Retain original outputs and reference
them from scoring records so future readers can audit each result.

## Adding a New Benchmark Run

1. Document new datasets in `datasets.md`.
2. Follow `protocol.md` for every experiment.
3. Store raw outputs and configurations under `runs/<run_id>/`.
4. Add one row to `results.csv` for each experiment run, including each repetition.
5. Do not change the protocol after seeing results.

## Results format

Use UTF-8 CSV with the existing header. Quote fields containing commas or newlines.
Use UTC ISO 8601 timestamps, seconds for runtime, and USD for estimated LLM cost.
Leave unavailable measurements blank and explain why in `notes`; zero means a
measured zero. For runs without an LLM, use `none` for provider/model and zero for
token usage and LLM cost. Do not add placeholder result rows.

Identifiers link a row to the frozen experiment plan (`experiment_id`), dataset
entry (`dataset_id`, `case_id`), and configuration (`configuration_id`).
`protocol_version` identifies the protocol revision; `mlcompass_commit` records
the full code commit. `repeat_index` is one-based. `artifacts_path` is relative
to this directory. Provider/model identifiers must identify the actual service
and model version used, with any unresolved alias documented in the run record.

Issue counts, status values, token accounting, and cost estimation are defined
in `protocol.md`. Keep individual repetitions, failures, and timeouts; summaries
may be added later without replacing raw run records.
