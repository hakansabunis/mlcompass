# Changelog

All notable changes to `ml-copilot` are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Planned for v0.2 (Faz 2)
- `ml-copilot audit <script>` — static training-script analysis
  (random seed, validation split, optimizer config, loss stability)
- `ml-copilot watch <script>` — live training monitor with plateau,
  overfitting, NaN, and divergence detection
- `ml-copilot compare <run-a> <run-b>` — run-to-run diff with
  LLM-explained hypotheses
- TensorBoard / W&B / plain-text log support
- Permission-gated config edits and training restarts

### Planned for v0.3 (Faz 3)
- `ml-copilot evaluate <results>` — post-training analysis
- Threshold optimization, confusion-matrix interpretation,
  hard-example surfacing

### Planned for v0.4 (Faz 4)
- `ml-copilot deploy --target <X>` — deployment readiness check
- Inference latency estimation, dependency consistency check,
  ONNX / TorchScript conversion advice

## [0.1.1] — 2026-05-29

### Fixed
- Add `pandas>=2.0.0` to runtime dependencies. The 0.1.0 wheel failed to
  import on a clean install because `tools.dataset` imports pandas at
  module load time but pandas was only present via the `dev` extras
  through `tbparse`. Caught by the TestPyPI fresh-venv smoke test
  before the bug reached production PyPI.

## [0.1.0] — 2026-05-29 (TestPyPI only — yanked)

First public release.

### Added
- `ml-copilot init <name>` — initialize a project context (`.mlcopilot/`)
  with metadata, decision log, dataset registry, run history, and cache
- `ml-copilot advise <data> [--target col] [--sample-rows N] [--no-llm]
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
- 66 passing tests across unit + integration layers

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
