# Changelog

All notable changes to `mlcompass` are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.3.1] — 2026-05-30

End-to-end pipeline release. Wraps Faz 2.2, Faz 3, Faz 4, Faz 5, and
the polish pass into a single tagged version. Every command on the
original roadmap is now in: ``init``, ``advise``, ``audit``,
``watch``, ``compare``, ``evaluate``, ``deploy``, ``status``. No
breaking changes from 0.2.0; users of the existing commands keep
their behaviour and gain four new commands plus optional LLM layers
across the board.

(0.3.0 was prepared but never published; this release supersedes it.)

### Added (Faz 5 — status command)
- `mlcompass status` — prints a structured snapshot of the active
  `.mlcompass/` project: name, created timestamp, default model;
  active dataset, project type, target column, current run, preferred
  models; a "command activity" tally aggregated from `advice.log`; and
  the most recent N decisions (default 5, configurable via `--recent`).
- Fresh project gracefully shows "no decisions recorded yet" + "nothing
  logged yet" placeholders; missing project exits non-zero with a
  pointer to `mlcompass init`.
- ui/status.py is a pure renderer — no LLM, no network — so the
  command is instant.
- 7 new tests covering fresh / populated project, command-activity
  tally, --recent cap, missing-project error, root --help discovery,
  and status --help option visibility. Full suite now at 361 passing.

This is the last command on the original roadmap. Every CLI surface
listed in v0.1's ARCHITECTURE.md §7 is now implemented.

### Added (Faz 3 — evaluate command)
- `mlcompass evaluate <results>` runs deterministic post-training
  analysis on a predictions table (CSV / Parquet / Excel / JSONL /
  JSON). Auto-detects `y_true`, `y_pred`, and `y_prob` columns from a
  small name-hint list (`label`, `target`, `pred`, `prob`, …) with
  explicit `--y-true / --y-pred / --y-prob` overrides.
- Task type is inferred from the data and the user can force it with
  `--task binary_classification | multiclass_classification | regression`.
- **Binary classification** outputs: accuracy / precision / recall / F1
  at threshold 0.5, AUC via the rank-based Wilcoxon–Mann–Whitney
  formula, 11-point threshold sweep with the best-F1 row starred,
  confusion matrix, hard examples ranked by `|y_true − y_prob|`, and
  imbalance / precision-recall lopsidedness warnings.
- **Multiclass** outputs: macro & weighted F1, per-class
  precision / recall / F1 / support, dense confusion matrix (rendered
  only for ≤ 8 labels), worst misclassified examples, and a "weak
  class" warning for classes with F1 < 0.3 and support ≥ 5.
- **Regression** outputs: MAE / RMSE / R², residual mean / std / min /
  max, top-k largest-residual rows, residual-bias warning when the
  mean residual is more than half the residual std.
- `--llm` opt-in interpreter (Faz 3c) runs Claude over the structured
  evaluation and returns `{assessment, strengths, weaknesses, next_steps}`
  rendered as a green panel + bullet lists. Pure reasoner (no tools);
  the deterministic analyzer already produced the numbers.
- `--hard-examples N` flag controls the top-k worst rows surfaced
  (default 5).
- `EvaluationError` and `EvaluateAgentError` map to clean red errors;
  evaluation failures exit 2, agent failures are non-fatal and print
  a single line.
- Pure pandas + numpy implementation; no scikit-learn dependency
  added.
- evaluate runs are persisted to `.mlcompass/advice.log` with the
  task, metrics, warnings, and (when `--llm` is set) the LLM
  interpretation.
- **Leakage-smell warning**: for any task, when the metrics look
  suspiciously perfect on ≥ 50 rows (binary AUC > 0.995 or accuracy
  > 0.99 or precision and recall both ≥ 0.99; multiclass accuracy
  > 0.99 or every class F1 ≥ 0.99; regression R² > 0.999) evaluate
  surfaces a clear warning naming the four usual causes — data
  leakage, train/test contamination, wrong column used as
  y_true / y_pred / y_prob, or scoring on the training set — so the
  user sanity-checks before believing the score.
- 51 new tests across four files (30 evaluation unit + 12 CLI
  integration + 5 LLM agent unit + 4 CLI `--llm` integration). Full
  suite now at 312 passing.

### Added (Faz 4 — deploy command)
- `mlcompass deploy <model_file>` runs a deployment-readiness check
  on a saved model file. Inspects metadata only — never loads the
  model (pickle exec, GPU init, and allocator surprises are footguns
  for an advisor).
- Format detection from magic bytes with extension fallback covers
  pytorch (`PK\\x03\\x04` Zip), pickle / joblib (`\\x80\\x04`),
  TensorFlow H5 (HDF5 signature), ONNX (protobuf), safetensors,
  GGUF, TFLite, and Core ML packages.
- **Pickle security warning** fires for raw `.pkl` / `.pickle`
  files, with a suggested switch to joblib / ONNX / safetensors.
- Size-class bucketing (small / medium / large / huge) with a
  pretty-printed file size.
- `--requirements <file>` reads `requirements.txt`,
  `pyproject.toml`, or `environment.yml` and reports pin status,
  unpinned dependency count, and which ML/DL packages are present.
  Unpinned-dep warning fires whenever any line lacks a version
  specifier.
- `--target {local, sagemaker, lambda, kubernetes, vertex}` (default
  `local`) layers in target-specific checks: Lambda flags models
  over its ~250 MB unzipped ceiling and warns about pickle cold
  starts; SageMaker / Vertex flag unknown formats; Kubernetes warns
  about unpinned deps as a "500-only-on-Friday" risk.
- Universal production checklist always renders monitoring nudges
  (inference latency, data drift, model drift, rollback path,
  logging / alerting) marked as `info` so they read as manual
  follow-ups, not as auto-passes.
- `--llm` opt-in advisor produces `{verdict, blockers[], next_steps[],
  rollout_strategy}` rendered as a green panel + bullet lists +
  cyan rollout panel. Same opt-in pattern as the other LLM layers.
- deploy runs are persisted to `.mlcompass/advice.log` with the
  format, size, target, warnings, full checklist, and (when --llm
  is set) the advisor's output.
- 42 new tests across four files (22 deploy unit + 11 CLI
  integration + 5 LLM agent unit + 4 CLI `--llm` integration). Full
  suite now at 354 passing.

With Faz 4 in place, every command on the original roadmap is
shipped: `init`, `advise`, `audit`, `watch`, `compare`, `evaluate`,
`deploy`.

## [0.2.1] — 2026-05-29

Quality-of-life follow-up to 0.2.0. The Faz 2.2 tasks land as one
patch: `watch` now reads TensorBoard event files and W&B local-run
directories alongside plain-text logs, and a new permission-gated
`--apply` flow can rewrite a config file from the diagnostician's
suggestions. No breaking changes.

### Added (Faz 2.2a — TensorBoard source)
- `mlcompass watch <path>` now accepts a TensorBoard event file
  (`events.out.tfevents.*`) or a directory that contains one, alongside
  the existing plain-text log support.
- `tools/tensorboard.py` — lazy `tbparse` import, raises
  `TensorBoardImportError` with a clear `pip install mlcompass[tensorboard]`
  hint when the optional dependency is missing.
- `tools/logs.py` gains `detect_source()` and `load_snapshots()` dispatcher
  helpers; `watch` uses them to route plain-text vs TensorBoard vs (planned)
  W&B without the detection rules needing to care which source produced
  the snapshots.
- `[tensorboard]` extras group: `pip install mlcompass[tensorboard]`.
- `--follow` is plain-text-only in v0.2; using it against a TensorBoard
  directory prints a clear yellow warning rather than crashing.
- 16 new tests (14 unit + 2 CLI integration). Full suite at 224 passing.

### Added (Faz 2.2b — W&B local source)
- `mlcompass watch <path>` now accepts a Weights & Biases local run
  directory, its `files/` subdirectory, or a `wandb-history.jsonl`
  file directly.
- `tools/wandb_local.py` reads the JSONL history, promotes `_step`
  to the snapshot's `step` field, surfaces `epoch` when present, and
  drops every other underscore-prefixed W&B internal (`_runtime`,
  `_timestamp`, `_wandb`, …) along with any non-numeric values.
- `detect_source()` recognises both layouts (run-dir + run-dir/files)
  and a direct `wandb-history.jsonl` / `wandb-summary.json` path.
- Defensive parser skips empty / malformed JSONL lines so a
  partially-written history file is still usable.
- No new dependency: the JSONL file is plain text.
- 15 new tests (12 unit + 1 dispatcher + 2 CLI integration). Full
  suite at 239 passing.

### Added (Faz 2.2c — permission-gated config edits)
- `mlcompass watch --llm --apply --config <file>` walks the
  diagnostician's `suggested_edits`, prompts the user once per edit,
  and rewrites the user's YAML or JSON config file in place.
- `-y / --yes` flag bypasses the prompt and applies every proposed
  edit (use with care).
- Watch diagnostician prompt now mentions an optional
  `suggested_edits` array in its JSON schema; each entry carries
  `key`, `current_value`, `proposed_value`, `rationale`. The required
  keys for the response (`diagnosis`, `summary`) are unchanged so
  older agent outputs keep parsing.
- `tools/config_edit.py`:
  - `ConfigEdit` dataclass plus `ApplyResult` (applied / rejected /
    skipped buckets and an optional backup path)
  - `load_config` / `write_config` accept both YAML and JSON
  - `apply_edits(path, edits, confirm_fn=...)` resolves dotted keys,
    skips edits whose key is missing or already matches, surfaces
    the live current value to the confirm callback (so drift is
    visible), and writes exactly one timestamped `.bak` file per
    apply session — only when at least one edit is accepted.
- `ui/config_edit.py` provides `make_console_confirm()` and
  `render_apply_summary()` for the terminal-side rendering.
- watch persists `applied_edits`, `rejected_edits`, and the backup
  path into the active `.mlcompass/advice.log` entry alongside the
  existing findings + diagnosis fields.
- 22 new tests (16 unit + 6 CLI integration). Full suite now at
  261 passing.


## [0.2.0] — 2026-05-29

Training-time tooling: three new modes (`audit`, `watch`, `compare`)
plus an opt-in LLM interpretation layer for each. Full pipeline is now
`init → advise → audit / watch / compare → (evaluate / deploy planned)`.

### Added (Faz 2a — audit mode)
- `mlcompass audit <script.py>` — pure-AST static analyzer for Python
  training scripts. No code is executed.
- Eight rules covering the most common ML pitfalls:
  - `seed` — error if no `torch.manual_seed`/`np.random.seed`/etc. set
  - `val_split` — warns when no validation split is detected, or split
    is implausibly small (<5%)
  - `optimizer` — errors on Adam-family + `momentum=` kwarg, infos SGD
    without momentum, warns on implausible learning rates
  - `loss_stability` — warns on `torch.log(x)` / `np.log(x)` without
    clamp/eps in the same expression
  - `dataloader` — warns when `torch.utils.data.DataLoader(...)` omits
    the `shuffle=` flag
  - `grad_clipping` — warns when the model uses RNN/Transformer
    modules but never calls `clip_grad_norm_`/`clip_grad_value_`
  - `eval_mode` — warns when a script calls `model.train()` but never
    `.eval()`
  - `batch_size` — info when an explicit `batch_size=` is implausibly
    small (<4) or huge (>4096)
- `--skip <rule>` flag (repeatable) to disable individual rules
- Rich terminal UI with severity-coloured findings table, summary
  counts, and suggested-fix footer
- Audit runs are persisted to the active `.mlcompass/` project as a
  decision entry plus a JSON line in `advice.log`
- 40 new tests (31 unit + 9 CLI integration) — full suite at 102 passing

### Added (Faz 2c — compare mode)
- `mlcompass compare <run-a> <run-b>` — diff two training runs.
  Accepts run IDs (resolved against the active `.mlcompass/runs/`) or
  direct directory paths.
- Run record format defined in `tools/runs.py`:
  - `config.yaml` with `name`, `created`, and a `config` mapping
  - `metrics.json` (`{"metrics": [...]}` or a bare list)
  - Optional `notes.md`
- Deterministic comparison surfaces:
  - Side-by-side header panel with each run's name, ID, and epoch count
  - Final-epoch metric table with deltas and per-metric winner column
  - Config-diff table with the keys that changed (including keys only
    present on one side)
  - Overall verdict (`a_better` / `b_better` / `mixed` / `inconclusive`)
    using lower-is-better / higher-is-better heuristics over metric names
- Compare runs are persisted to the active project as a decision entry
  plus a JSON line in `advice.log`
- 25 new tests (17 unit + 8 CLI integration). Full suite at 127 passing.

### Added (Faz 2b — watch mode)
- `mlcompass watch <log_file>` — monitor a plain-text training log for
  the four most common training-time pathologies.
- Lenient log parser (`tools/logs.py`) accepts `key=value`, `key: value`,
  and `key  value` pairs across the dominant print-style and Keras-style
  formats. Handles scientific notation, negative numbers, NaN, and Inf.
  Multi-line "Epoch N" entries are merged into a single snapshot.
- Four pure-function detectors in `tools/anomaly.py`:
  - `nan` — error if any loss-like metric in the latest snapshot is
    NaN or ±Inf
  - `divergence` — error when train loss jumps ≥10× between
    consecutive snapshots
  - `plateau` — warning when the primary loss (val_loss preferred,
    train_loss fallback) is flat across the last 5 snapshots
  - `overfitting` — warning when train loss is falling, val loss is
    rising, and the absolute train/val gap exceeds 0.05
- `--follow / -f` flag tails the log file and surfaces only newly-detected
  findings as fresh epochs arrive; `--poll-interval` is configurable
- Rich rendering (`ui/watch.py`):
  - Overview panel (log path, snapshot count, last epoch, per-severity
    finding counts)
  - Recent-metrics table that adapts its columns to whichever metrics
    the last snapshots actually carried
  - Findings table with severity colouring and top-3 suggested-fix footer
- Watch runs are persisted to the active `.mlcompass/` project as a
  decision entry plus a JSON line in `advice.log`
- 48 new tests (20 log parser + 19 anomaly detector + 9 CLI integration).
  Full suite now at 175 passing.

### Added (Faz 2.1 — optional LLM layers)
- `--llm` opt-in flag on `audit`, `watch`, and `compare`. When set
  (and `ANTHROPIC_API_KEY` is present) Claude runs on top of the
  deterministic output and adds interpretation.
- `agents/audit.py` — **prioritizer**: ranks findings by blast radius
  and writes a one-paragraph synthesis.
- `agents/watch.py` — **diagnostician**: for each anomaly, hypothesises
  the root cause and proposes a concrete next action with a confidence
  tag ("high" / "medium" / "low").
- `agents/compare.py` — **hypothesizer**: explains *why* the winning
  run won, identifies the key config factors driving the outcome, and
  proposes the next experiment.
- `agents/_common.py` — shared lenient JSON parser that handles
  optional markdown fences, validates the response is a dict, and
  enforces required top-level keys.
- UI extensions (`render_audit_priorities`, `render_watch_diagnosis`,
  `render_compare_hypothesis`) render each LLM output as a rich
  table + panel.
- Each LLM result is written into `.mlcompass/advice.log` alongside
  the deterministic output, so the project history captures both.
- Graceful degradation: missing API key, empty input (e.g. no
  findings for audit / watch to prioritize), and malformed agent
  responses all print a clear, non-fatal message rather than
  exiting non-zero.
- 33 new tests (21 unit + 12 CLI integration). Full suite now at
  208 passing.

### Planned for v0.3 (Faz 3)
- `mlcompass evaluate <results>` — post-training analysis
- Threshold optimization, confusion-matrix interpretation,
  hard-example surfacing

### Planned for v0.4 (Faz 4)
- `mlcompass deploy --target <X>` — deployment readiness check
- Inference latency estimation, dependency consistency check,
  ONNX / TorchScript conversion advice

## [0.1.0] — 2026-05-29

First public release on PyPI.

### Added
- `mlcompass init <name>` — initialize a project context (`.mlcompass/`)
  with metadata, decision log, dataset registry, run history, and cache
- `mlcompass advise <data> [--target col] [--sample-rows N] [--no-llm]
  [--model NAME]` — analyse a dataset and produce model + feature
  engineering + pitfall recommendations
- Project context infrastructure (`ProjectContext`):
  - `init()`, `load()` (walks up like git's repo discovery)
  - `read_context()`, `write_context()`, `append_decision()`
  - `register_dataset()` with mtime-based fingerprinting
- Dataset analyzer (`tools.dataset.analyze_dataset`):
  - CSV / Parquet / Excel / JSON Lines / JSON loading
  - Per-column classification: numeric, categorical, datetime,
    boolean, text
  - Numeric stats (mean/std/quartiles/min/max) + IQR and Z-score
    outlier counts
  - Categorical cardinality + top-5 values
  - Datetime min/max ranges
  - Target detection (cascading: explicit → high → medium → low
    fallback → none)
  - Task inference (regression / binary / multiclass / unknown)
  - Class balance and warnings (missing data, high cardinality,
    class imbalance)
- Advisor agent (`agents.advise.get_recommendation`):
  - Senior-data-scientist persona, JSON-only output contract
  - Tolerant response parser (handles markdown fences, validates
    shape, raises `AdvisorParseError` on bad responses)
- Rich terminal UI for both analysis and recommendation
- Three deterministic example datasets (titanic, house prices,
  customer churn) with a documented generator script
- 62 passing tests across unit + integration layers

### Architecture
- Single CLI binary using Click
- agentlite-py as the agent backbone
- Module layout: `cli.py`, `context.py`, `agents/`, `tools/`, `ui/`
- UTF-8 stdio reconfiguration for Windows non-UTF-8 codepages

### Known limitations
- Binary integer targets read from CSV are typed as numeric, not boolean
- Datetime columns saved as ISO strings in CSV are read as text;
  use Parquet or specify `parse_dates` if datetime semantics matter
- Multi-language project support not yet planned
- No hosted version — local-only at v0.x

### Note on naming
mlcompass was prototyped under the working name `ml-copilot`. PyPI's
"too similar to existing project" guard blocked that name on the final
upload (Microsoft maintains the `mlcopilot` package as a different
Jupyter-focused tool), so the project was renamed before its first
release. No `ml-copilot` package was ever published to production PyPI.
