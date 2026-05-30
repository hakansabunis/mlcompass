# Changelog

All notable changes to `mlcompass` are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

### Planned for v0.2 (remaining Faz 2)
- `mlcompass watch <script>` — live training monitor with plateau,
  overfitting, NaN, and divergence detection
- TensorBoard / W&B / plain-text log support
- Permission-gated config edits and training restarts
- Optional LLM auditor layer that explains and prioritizes findings
- Optional LLM compare layer ("Run B is better because…" hypothesis)

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
