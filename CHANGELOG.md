# Changelog

All notable changes to `mlcompass` are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

### Planned for the rest of v0.2
- Permission-gated config edits and training restarts

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
