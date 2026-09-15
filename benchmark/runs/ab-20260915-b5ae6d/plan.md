# A/B experiment plan — ab-20260915-b5ae6d

Frozen at 2026-09-15T11:54:43Z (UTC), before execution.

- Protocol: ab_protocol.md, version A/B 1.1
- Dataset registry: ground_truth.json v1.1, frozen 2026-09-15 (used here for the pinned files, their targets and their task types only — the case list is not scored by this battery)
- mlcompass commit: c8754cc351900c750e4f91ff925cb8b3953fbf6d
- Tree dirty at planning time: True
- Python 3.13.3 on Windows-11-10.0.26200-SP0
- Repetitions per cell: 1; seeds [20260915]
- Timeouts: model call 900s, emitted script 600s, holdout scoring 300s
- Max completion tokens: 4096

## Generation settings (section 5, frozen by §9 A3)

- Temperature: 1.0 on every arm and every provider, sent explicitly
  in the request body rather than left to the provider default, and recorded in
  every run record.
- A provider that rejects the parameter by name has the rejection recorded and
  its `temperature_used` reads `unknown`. The row is kept; it never claims a
  temperature the provider did not honour. A failure that does not name the
  parameter is recorded as a failure, not read as a rejection.

## Split (section 3, frozen by §9 A2)

- Holdout fraction: 0.25
- Stratified on the target for classification tasks: True
- Split once per (dataset, seed) with `sklearn.model_selection.train_test_split`,
  `shuffle=True`, `random_state=<seed>`. Every arm and every repetition at that
  seed are scored on the identical holdout.
- Both values are the protocol's, read back out of ab_protocol.md §3 by
  `verify_protocol_constants` before any cell runs. A harness that disagreed
  with the document would abort rather than split.

## Arms (section 2, three since §9 A1)

- `control`: train.csv path, target column, column listing, write-a-script task.
- `advise`: byte-identical prompt plus an appended block carrying the verbatim
  stdout of `mlcompass advise` on the same train.csv. The advise prompt is
  constructed as `control + block`, so those two arms cannot differ elsewhere.
- `advise+audit`: the `advise` arm's turn, byte for byte, then one revision
  round. `mlcompass audit` is run on the script *that arm's own* first turn
  produced, its verbatim stdout goes back to the same model in a second user
  message, and the revised script is the one scored. No arm ever sees another
  arm's output. An audit that finds nothing still produces the round, with an
  empty findings block.

## Models (section 5 — reported per model, never pooled)

- `ollama-qwen2.5-7b`: `qwen2.5:7b` via `openai` at http://localhost:11434/v1 — local, free, no credential.

## Cells

| dataset | arm | model | repeats | seeds | metric |
| --- | --- | --- | --- | --- | --- |
| openml-1464 | control | ollama-qwen2.5-7b | 1 | 20260915 | roc_auc |
| openml-1464 | advise | ollama-qwen2.5-7b | 1 | 20260915 | roc_auc |
| openml-1464 | advise+audit | ollama-qwen2.5-7b | 1 | 20260915 | roc_auc |

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
