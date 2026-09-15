# A/B experiment plan — ab-20260915-a97d3f

Frozen at 2026-09-15T11:18:37Z (UTC), before execution.

- Protocol: ab_protocol.md, version A/B 1.0
- Dataset registry: ground_truth.json v1.0, frozen 2026-09-15 (used here for the pinned files, their targets and their task types only — the case list is not scored by this battery)
- mlcompass commit: 7a2414e96be210df4abbfe68f43b9f905ddd9dd9
- Tree dirty at planning time: True
- Python 3.13.3 on Windows-11-10.0.26200-SP0
- Repetitions per cell: 1; seeds [20260915]
- Timeouts: model call 900s, emitted script 600s, holdout scoring 300s
- Max completion tokens: 4096

## Split (section 3)

- Holdout fraction: 0.25
- Stratified on the target for classification tasks: True
- Split once per (dataset, seed) with `sklearn.model_selection.train_test_split`,
  `shuffle=True`, `random_state=<seed>`. Both arms and every repetition at that
  seed are scored on the identical holdout.
- The fraction and the stratification choice are not fixed by ab_protocol.md.
  They are fixed here, before execution, and recorded so a later run cannot
  quietly use different ones.

## Arms (section 2)

- `control`: train.csv path, target column, column listing, write-a-script task.
- `treatment`: byte-identical prompt plus an appended block carrying the verbatim
  stdout of `mlcompass advise` on the same train.csv. The treatment prompt is
  constructed as `control + block`, so the arms cannot differ anywhere else.
- `mlcompass audit` takes a training script and no script exists at
  prompt-construction time, so its block is present and empty. ab_protocol.md
  section 2 does not specify audit's input for a cell that has no script; this
  harness does not invent one.

## Models (section 5 — reported per model, never pooled)

- `ollama-qwen2.5-7b`: `qwen2.5:7b` via `openai` at http://localhost:11434/v1 — local, free, no credential.

## Cells

| dataset | arm | model | repeats | seeds | metric |
| --- | --- | --- | --- | --- | --- |
| openml-1464 | control | ollama-qwen2.5-7b | 1 | 20260915 | roc_auc |
| openml-1464 | treatment | ollama-qwen2.5-7b | 1 | 20260915 | roc_auc |

## Scoring (sections 3 and 4)

- `holdout_score`: the saved model loaded by the harness and predicted on
  holdout.csv. ROC AUC for binary, macro F1 for multiclass, R² for regression.
  A script that saved nothing loadable scores blank with a recorded reason;
  no number is substituted.
- `process_defects`: the six-item checklist, a frozen deterministic function of
  the emitted source. Never scored by a model.
- `runs_at_all`: the emitted script's exit status.

## Execution order

Sequential, dataset-major then arm then model. Every run executes the emitted
script as a subprocess in a fresh temp workspace holding train.csv only, under
a wall-clock timeout, with holdout.csv outside that workspace.
