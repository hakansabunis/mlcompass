# Changelog

All notable changes to `mlcompass` are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Planned for v0.2 (Faz 2)
- `mlcompass audit <script>` — static training-script analysis
  (random seed, validation split, optimizer config, loss stability)
- `mlcompass watch <script>` — live training monitor with plateau,
  overfitting, NaN, and divergence detection
- `mlcompass compare <run-a> <run-b>` — run-to-run diff with
  LLM-explained hypotheses
- TensorBoard / W&B / plain-text log support
- Permission-gated config edits and training restarts

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
