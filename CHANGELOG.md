# Changelog

All notable changes to `mlcompass` are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.7.2] — 2026-05-31

**Field-test patch release.** A full 8-tool end-to-end pipeline run
from Claude Code's MCP integration on the Palmer Penguins dataset
(via `mlcompass-mcp`) surfaced two real bugs and one target-name
gap. This release closes all three.

### Fixed
- **#FT4-1 — multiclass evaluate binary fallback** (`tools/evaluation.py`).
  Pre-v0.7.2 `_infer_task` short-circuited to `binary_classification`
  the moment a `y_prob` column was present, even when `y_true` had
  3+ distinct labels. On Penguins, this surfaced as a hard error:
  `Binary task expected 2 distinct labels, got 3`. The fix: rank
  `y_true.nunique() > 2 → multiclass` above the y_prob heuristic.
  Numeric binary 0/1 with a probability column still infers binary
  (regression guard).
- **#FT4-2 — MCP tools stateless** (`mcp_server.py`). The CLI commands
  have always written to `.mlcompass/context.json` and `advice.log`,
  but the MCP tools didn't — so `mlcompass_status` came back empty
  even after a full pipeline run through MCP. Now `init`, `advise`,
  `audit`, `watch`, `compare`, `evaluate`, and `deploy` all append
  a structured decision entry + an advice.log line when an active
  project is found. Missing project ⇒ silent skip (no crash). Every
  ledger write is best-effort via `contextlib.suppress`, so a write
  failure can never cascade into a tool-call failure.

### Improved — `advise` target detection (Field Test #4)
- **Penguins / multiclass classification names** join the high-
  confidence list: `species`, `class_label`, `category_label`,
  `digit`, `genre`, `sentiment`, `intent`, `language`. Picked
  specifically because they cover the canonical multiclass labels
  across image / NLP / biology Kaggle competitions (Penguins, MNIST,
  movie genres, sentiment analysis, intent classification, language
  detection).

### Tests
- 8 new regression tests:
  - 3 `_infer_task` multiclass-with-y_prob path (Penguins, string
    labels, binary regression guard).
  - 2 new target-name regressions (Penguins `species`, MNIST `digit`).
  - 3 MCP persistence integration tests: `advise` writes decisions,
    no-project silent skip, full pipeline (`advise` + `audit`)
    populates `status` correctly.
- Three existing `test_status_*` tests updated to expect the v0.7.2
  init-decision marker that `mlcompass_init` now appends.
- `mlcompass_compare` summary helper hardened against string-vs-dict
  verdict shapes (caught by the existing compare test).
- Full suite: **528 passing**, 2 skipped (was 520 + 8 new).
- `ruff check` / `ruff format --check` clean.
- `mypy --strict` clean across **51** source files.

### Verification on the live Penguins pipeline
After this patch, the Field Test #4 demo path through Claude Code's
MCP now succeeds at every step:
- `mlcompass_evaluate` on `predictions.csv` (Penguins multiclass +
  `y_prob` column) infers multiclass automatically — no `task=`
  parameter needed.
- `mlcompass_status` reflects every MCP tool call made during the
  session: `init`, `advise`, `audit`, `watch`, `compare`, `evaluate`,
  `deploy` all show up in `command_counts` and `decisions`.
- `mlcompass_advise` on a dataset with a `species` column auto-
  detects it as the target with high confidence.

## [0.7.1] — 2026-05-31

**Field-test patch release.** A live test run from Claude Code's MCP
integration on the Kaggle Titanic dataset surfaced that the v0.7
target-name heuristics — beefed up with regression names during the
Ames House Prices field test — was still missing the *flagship*
binary classification targets every Kaggle starter dataset uses.
The analyzer was falling back to the last-column rule and picking
``Embarked`` as the target on Titanic when ``Survived`` was right
there. This release closes that gap.

### Improved — `advise` target detection (Field Test #3)
- **Survived (Titanic)** and 11 more canonical binary names join
  ``TARGET_NAME_HINTS["high_confidence"]``: ``survived``,
  ``is_survived``, ``purchased``, ``is_purchase``, ``is_purchased``,
  ``clicked``, ``is_clicked``, ``converted``, ``is_converted``,
  ``accepted``, ``approved``, ``winner``. Picked specifically because
  they show up across the most-used Kaggle competitions
  (Titanic, Customer Churn, Click-Through Rate, Marketing
  Conversions, Loan Approval, …).
- Three softer signals land in ``medium_confidence``: ``engagement``,
  ``subscribed``, ``active``.

### Why this happened
The original v0.1 target-name list was tiny (``target``, ``label``,
``y``, ``churn``, ``fraud``, ``default``). v0.7 fixed the regression
gap (FT#2 / Ames). v0.7.1 fixes the classification gap (FT#3 /
Titanic) — same kind of fix, different task type. Going forward,
new field tests will keep extending the lists, because the cost of
a false-negative target detection is high: the user gets a worse
analysis than they would from `--target <name>` and has no way to
tell that's why.

### Tests
- 6 new regression tests:
  - `Survived` auto-detected with high confidence.
  - `purchased`, `clicked` flagship binary names.
  - `is_converted` (the `is_*` prefix variant).
  - `engagement` lands in medium confidence (softer signal).
  - Regression guard: existing `churn`, `fraud`, `saleprice`,
    `is_fraud` still detected high-confidence after the list grew.
- Full suite: **520 passing**, 2 skipped (was 514 + 6 new).
- `ruff check` / `ruff format --check` clean.
- `mypy --strict` clean across 51 source files.

### Verification on live Titanic CSV
After this patch, the Claude Code MCP demo path (download titanic.csv
→ `mlcompass_advise(dataset_path="titanic.csv")`) now reports:
- Target: `Survived` (high confidence)
- Task: binary classification, ~38% positive
…instead of the misleading "Embarked / low" fallback.

## [0.7.0] — 2026-05-31

The headline of v0.7 is **automatic leakage investigation**. When
`mlcompass evaluate` sees a suspiciously perfect metric (AUC > 0.995,
accuracy > 0.99, R² > 0.999), it now **runs a deterministic
investigator** that gathers structured evidence about which columns
might be the source — per-feature correlations against the target,
the exact y_pred=y_true match rate, and a candidate-leak list. With
`--llm`, an anti-hallucination Claude agent then narrates the
evidence into a verdict + recommended manual checks. The
investigator is forbidden from inventing columns or proposing code
patches — it cites only what's in the evidence dict, or admits
``cannot_determine``.

This release also closes the four field-test findings the Ames House
Prices dry run surfaced.

### Added — Faz 9d / 9e / 9f (leakage auto-investigation)
- New `src/mlcompass/tools/leakage.py` — pure pandas + numpy
  evidence collector. Computes:
  - **Target ↔ feature correlations** (top 10 by |corr|). Uses
    Pearson for numeric targets (binary 0/1 ints included), and
    rank-based Pearson for string class labels. For numeric targets
    we now also compute the rank-based score and **report whichever
    is stronger** — that's what catches monotone leaks like a
    `log_price` column that's just `log(target)` (Pearson ~0.94,
    Spearman = 1.00).
  - **Exact y_pred == y_true match rate** — over 95% on a non-
    trivial task is a smoking gun.
  - **Candidate leak columns** — any feature whose |correlation|
    with the target ≥ 0.99.
  - **Sample-size trustworthiness flag** — below 50 rows the agent
    is told to downgrade its confidence.
- New `src/mlcompass/agents/leakage_investigator.py` — Claude-driven
  narrator with a strict anti-hallucination contract:
  - "Cite ONLY items present in the evidence dict."
  - "If candidate_leak_columns is empty AND perfect_match_rate < 0.95,
    say `cannot_determine`. Do NOT speculate."
  - "Recommendations must be MANUAL checks. NEVER propose code
    patches."
  - Output schema clamped at the agent boundary (enum verdict +
    confidence) so a stray hallucinated value can't break the UI.
- `evaluate()` auto-attaches `result["leakage_investigation"]`
  whenever the smell threshold fires (no flag needed — the
  deterministic evidence is always free of charge). `--llm` then
  chains the investigator agent on top.
- New "🔬 Leakage investigation — evidence" panel (red border)
  + "🔍 Leakage investigator (Claude)" panel (magenta border).
- 11 leakage-tool tests covering numeric binary, string binary,
  regression with monotone leaks, sample-size guard, K=10 cap, and
  the schema contract.

### Improved — `advise` (Ames Field Test #2 patches)
- **FT#2-1**: Regression target name hints. `TARGET_NAME_HINTS`
  extended with `saleprice`, `sale_price`, `price`, `saleamount`,
  `sale_amount` (high-confidence) and `amount`, `value`, `score`,
  `revenue`, `cost`, `salary`, `rating` (medium). Ames-style
  regression targets now auto-detect.
- **FT#2-2**: Year/Yr columns escape the "2 distinct values →
  categorical" v0.6.1 rule. Columns whose name matches `year`,
  `yr`, `_yr_`, `_year_` AND whose min lands in the plausible
  1800-2200 band stay numeric — `Yr Sold` 2006-2010 is temporal,
  not nominal.
- **FT#2-3**: Sparse numeric columns. When > 50% of values are zero
  (Ames `Open Porch SF`, `Pool Area`, …) the analyzer flags the
  column with `sparse: True`, skips the otherwise-misleading IQR
  outlier count, and adds a "consider binarising or zero-inflated
  model" warning.

### Tests
- 14 new dataset analyzer tests: 3 target-hint regressions (FT#2-1),
  3 year-column heuristic edge cases (FT#2-2), 2 sparse-column
  scenarios (FT#2-3), 6 v0.6.1 regressions kept green.
- 11 leakage-tool tests (Faz 9d).
- `with_api_key` fixture in `test_cli_evaluate_llm.py` auto-stubs
  the leakage investigator so existing `evaluate --llm` tests
  don't accidentally hit the real Anthropic API.
- Full suite: **514 passing**, 2 skipped (was 495 + 19 new).
- `ruff check` / `ruff format --check` clean.
- `mypy --strict` clean across **51** source files (was 49).

## [0.6.1] — 2026-05-31

**Field-test patch release.** A dry run against the Kaggle Telco
Customer Churn dataset surfaced two real bugs and four UX rough
edges; this release closes all six. No CLI surface or on-disk
project layout changes, no breaking changes — 0.6.0 users upgrade in
place.

### Fixed
- **Status panel "mlcompass ver: —"** (field-test bug #7). `init`
  writes `mlcompass_version: <ver>` into `project.yaml` but
  `ui/status.py` was looking it up under the legacy
  `ml_compass_version` (with the underscore) — a leftover key from
  the v0.4 ml-copilot → mlcompass rename. The panel now reads the
  current key first and falls back through both legacy spellings.
- **Agent CLI no-API-key crash** (field-test bug #6). The `api`
  backend used to hand control to `anthropic` and crash one turn in
  with a noisy `Could not resolve authentication method` error.
  The CLI now detects the missing `ANTHROPIC_API_KEY` up front,
  prints a clean usage hint (with both the env-var instructions and
  the `--backend claude-code` alternative), and exits 2. The
  `claude-code` backend skips this check — it has its own auth flow
  inside the Claude Code CLI.

### Improved — `advise` dataset heuristics
- **Numeric-with-dirty-strings columns** (field-test UX #1). Telco's
  `TotalCharges` column should be numeric but a few rows contain
  empty strings, so pandas drops it to `object`. The analyzer now
  detects this pattern (≥95% of values parse as float) and adds a
  "looks numeric but contain non-numeric values; clean before
  training" warning that names the offending columns.
- **Binary 0/1 numeric columns force-categorical** (field-test UX
  #2). Pre-v0.6.1, `SeniorCitizen`-style 0/1 columns were
  classified as numeric and the IQR outlier counter would report
  nonsense numbers (e.g. "1142 IQR outliers" on a vector of 0s and
  1s). Columns with exactly two distinct numeric values are now
  classified as categorical with a cardinality of 2.
- **Unique-per-row identifier columns flagged** (field-test UX #3).
  A text column whose cardinality equals the row count is almost
  always an ID; using it as a feature lets the model memorise rows.
  The analyzer now adds a "look like unique-per-row identifiers
  (`customerID`, …); drop before training" warning.

### Improved — `optimize` suggestions
- **Domain-aware soft caps for well-known hyperparameters** (field-
  test UX #5). The Telco run surfaced an explore-step suggestion of
  `dropout=0.95` — beyond what's actually trainable. The suggester
  now consults a small table of soft caps when the user hasn't
  supplied `--constraints`: dropout ≤ 0.8, weight_decay ≤ 1.0,
  momentum ≤ 0.99, lr ∈ [1e-7, 10]. Substring-matched, so
  `learning_rate`, `decoder_dropout` etc. all hit the right cap.
  Explicit `--constraints` always override the soft caps.

### Tests
- 14 new tests:
  - 2 status version-key regressions (current key + legacy fallback).
  - 2 agent-CLI missing-key tests (api backend exits 2,
    claude-code backend runs unaffected).
  - 6 dataset analyzer tests (numeric-with-dirty-strings positive +
    negative, binary-numeric, three-value-numeric stays numeric,
    unique-per-row ID positive + negative).
  - 4 optimize perturbation soft-cap tests (dropout cap, lr cap,
    user-bounds override, unknown name unrestricted).
- Full suite: 495 passing, 2 skipped (was 481 + 14 new).
- `ruff check` / `ruff format --check` clean; `mypy --strict` clean
  across 49 source files.

## [0.6.0] — 2026-05-31

Three new capabilities ship together: **post-deploy drift detection
(`monitor`)**, **hyperparameter optimization (`optimize`)**, and
**cross-session agent memory** so the self-driving agent from v0.5
now remembers prior decisions and prior runs across invocations. The
CLI grows from nine to eleven commands; v0.5 users upgrade in place
with no breaking changes.

### Added — Faz 8a (`monitor`)
- New `src/mlcompass/tools/drift.py` — pure-numpy drift detector. PSI
  (Population Stability Index) as the primary score, with KS test for
  numeric features and chi-square for categoricals as corroborating
  signals. Quantile-based binning with epsilon guards for constant
  reference distributions; asymptotic Kolmogorov and incomplete-gamma
  survival functions implemented inline so we don't pull in scipy.
- Industry-standard PSI thresholds: `< 0.1` stable, `0.1–0.2`
  moderate, `≥ 0.2` major. Aggregate verdict (`stable` /
  `moderate_drift` / `major_drift`) with a `retrain_recommended` flag.
- pandas 2.x `StringDtype` is now classified as categorical (caught
  during smoke testing — the legacy `is_object_dtype` check missed
  it). Also forward-proofed for pandas 4: `is_categorical_dtype`
  replaced with `isinstance(dtype, pd.CategoricalDtype)`.
- New `mlcompass monitor <reference> <current>` CLI: 9th-listed
  subcommand. Exits **1** on major drift so CI / cron pipelines can
  gate on it. Flags: `--features`, `--bins`, `--top`, `--llm`,
  `--model`.
- Optional `--llm` interpreter (`agents/monitor.py`): hands the
  structured drift report to an `agentlite`-backed Claude prompt that
  returns `{headline, likely_cause, next_steps}`.
- 12 tests for the drift tool layer + 9 CLI tests, including the
  StringDtype regression test, constant-reference no-crash test, and
  the leakage-style "small sample size" warning path.

### Added — Faz 8b (`optimize`)
- New `src/mlcompass/tools/optimize.py` — HPO sub-agent that reads a
  run history (`<project>/runs/<id>/` directories) and produces a
  leaderboard, per-hyperparameter rank-correlation sensitivity, and
  a portfolio of suggested next configs. Pure numpy.
- **Multiplicative perturbation** for strictly-positive hyper-
  parameters: a leader with `lr=0.0001` now steps to `3.16e-5` /
  `3.16e-4` (half-decade moves) instead of subtracting an absolute
  delta and producing nonsensical negative learning rates. User-
  supplied `--constraints` switch back to additive bounds-clamping.
- Spearman-style rank correlation with average ties, implemented in
  pure numpy. Integer-typed hyperparameters stay integer after
  perturbation.
- New `mlcompass optimize --metric <name>` CLI: 10th-listed
  subcommand. Flags: `--runs-dir`, `--direction max|min`, `--top`,
  `--suggestions`, `--constraints lr:lo-hi,batch_size:lo-hi`,
  `--llm`, `--model`. Defaults `--runs-dir` to `<project>/runs/`
  when an mlcompass project is active.
- Optional `--llm` strategist (`agents/optimize.py`): returns
  `{headline, pattern, next_plan}` synthesising the run history.
- 18 tests for the optimize tool layer + 9 CLI tests, including a
  monotone-history fixture, min-direction sanity, constraint parsing,
  the "positive lr stays positive" regression test.

### Added — Faz 8c (agent memory)
- New `src/mlcompass/agent/memory.py` — two complementary memory
  sources stitched into a single `MemoryBlock` the orchestrator
  prepends to every new agent run:
  - **Project decisions** from `.mlcompass/context.json` (latest
    10 by default, configurable).
  - **Prior agent runs** from `.mlcompass/agent_runs/<id>/transcript
    .jsonl`. Transcripts are compressed by an `agentlite`-driven
    summariser sub-agent (100-200 word narrative covering the task,
    tools called, key findings, final answer, follow-ups) — the
    user's own agentlite-py library finally gets a meaningful role
    inside mlcompass.
- New `--resume <run-id>` flag on `mlcompass agent` — resumes from
  a prior run by id (full or timestamp prefix). The orchestrator
  loads the transcript, summarises it, and seeds the new run with
  that context plus the decisions log.
- New `--no-memory` flag — "fresh slate" mode that bypasses the
  default cross-session injection. Useful when you want the agent
  to ignore prior history (e.g. a completely new project direction).
- `MemoryBlock.as_prompt_prefix()` renders to a structured XML-like
  block (`<memory_headline>`, `<recent_decisions>`, `<prior_agent_run
  id='...'>`) so the model can parse the memory cleanly from the
  user task.
- The memory block is written as the FIRST transcript step of the
  new run, so the audit trail records what context the agent saw.
- All memory paths are best-effort: missing context.json, missing
  transcript, unparseable JSON, or missing agentlite-py each fall
  through to graceful "no memory" without crashing.
- 20 new tests covering decisions reader, transcript I/O,
  truncation, run-dir discovery (by full id and timestamp prefix),
  high-level memory builders, and the agentlite-missing fallback.

### Validation
- `pytest tests/`                   → **481 passing**, 2 skipped
                                        (was 412 + 69 new).
- `ruff check src tests`            → 0 errors.
- `ruff format --check src tests`   → 89 files clean.
- `mypy --strict src/mlcompass`     → 0 errors across **49** source
                                        files (was 42).

## [0.5.0] — 2026-05-30

mlcompass becomes self-driving. The new `mlcompass agent "<task>"`
command hands the eight pipeline tools to a Claude agent that picks
the right call sequence, streams each step to the terminal, and
writes a transcript you can audit later. Two backends ship together:
`api` (default — universal, only needs `ANTHROPIC_API_KEY`) and
`claude-code` (routes through your local Claude Code CLI via
Anthropic's official Agent SDK). The CLI and MCP surfaces from
v0.3.x / v0.4.0 stay byte-identical; everything new is additive.

### Added — Faz 7 (agent layer)
- New `src/mlcompass/agent/` package:
  - `tools.py` — single registry mapping each of the eight
    `mlcompass_*` tools to a JSON Schema + dispatcher. Both backends
    read from this registry, so fixing a tool once propagates
    everywhere (CLI's `--llm` modes, MCP server, agent backends).
  - `backends/_common.py` — `AgentBackend` protocol, `AgentStep`,
    `AgentResult`, and the `PermissionCallback` / `StepCallback`
    types every backend consumes.
  - `backends/anthropic_api.py` — universal backend: hand-rolled
    tool-use loop on `anthropic.Anthropic().messages.create`, with
    streaming + permission gating + max-turns safety cap.
  - `backends/claude_code.py` — opt-in backend: wraps each tool with
    `claude_agent_sdk.tool`, bundles them via `create_sdk_mcp_server`,
    and iterates `claude_agent_sdk.query` (async). Requires the
    `claude` CLI on PATH.
  - `orchestrator.py` — `run_agent(task, backend=..., ...)` entry
    point that picks a backend, wires the streaming UI + transcript
    writer, and returns an `AgentRunSummary`.
  - `transcript.py` — per-run journal under
    `.mlcompass/agent_runs/<id>/transcript.jsonl` + a
    human-readable `summary.md`. Giant tool results are truncated on
    disk so the journal stays grep-friendly.
  - `ui.py` — rich streaming renderer + a `Confirm.ask`-backed
    permission callback for the CLI.
  - `_system_prompt.py` — shared system prompt; six operating rules
    (plan briefly, one tool at a time, read results carefully, don't
    invent paths, respect existing project context, stop when done).
- New `mlcompass agent "<task>"` CLI subcommand with `--backend`,
  `--project-path`, `--model`, `--max-turns`, `--auto-approve` flags.
  Exits 0 on convergence, 1 on failure (max-turns or backend error).
- New optional dependency groups: `mlcompass[agent]` pulls
  `anthropic>=0.50.0`; `mlcompass[agent-claude-code]` pulls
  `claude-agent-sdk>=0.2.0`. Both are guarded with clean ImportError
  messages pointing at the install instructions.

### Design constraints
- The agent surface has **no `--llm` knob** — it's LLM-first by
  design. The CLI's per-tool `--llm` reasoning modes (audit / watch /
  compare / evaluate / deploy) stay where they are for non-agentic
  scripted use.
- The default model is `claude-sonnet-4-5` (cheaper for routing).
  Override via `--model` or the `MLCOMPASS_AGENT_MODEL` env var.
- The only mutating tool is `mlcompass_init`; the agent's permission
  gate fires only on that unless `--auto-approve` is passed. Six
  read/compute tools auto-allow.
- The orchestrator falls back to `<project_path>/.mlcompass-agent/`
  when no `.mlcompass/` exists, so the audit trail is preserved
  without silently creating a half-initialised project.
- Both backends share one tool registry → a fix lands in both at
  once, plus the MCP server stays in sync via its own re-use.

### Tests
- 28 new tests:
  - 8 for the shared tool registry (eight tools, mutation flag,
    schema invariants, dispatcher round-trip, error envelopes).
  - 4 for the transcript writer (unique run-id, JSONL append, giant
    payload truncation, summary metadata).
  - 5 for the Anthropic API backend (fake-client text path, tool
    round-trip, permission denial, max-turns, API error).
  - 4 for the claude-code backend (mocked SDK text path, tool-use
    round-trip via the namespaced `mcp__mlcompass__*` names, missing
    dependency error message, SDK exception → typed failure).
  - 7 for the CLI subcommand (default backend, alt backend,
    `--auto-approve`, `--max-turns`, non-zero exit on failure,
    `--help` discovery for the agent subcommand and root).
- Full suite: 412 passing, 2 skipped (was 384 + 28 new).
- `ruff check src tests` → 0 errors.
- `mypy --strict src/mlcompass` → 0 errors across **42** source files
  (was 32).

## [0.4.0] — 2026-05-30

mlcompass goes beyond a CLI: every command is now reachable from
**Claude Desktop, Claude Code, Cursor, and any other MCP-capable
client** via the new `mlcompass-mcp` server. Same eight tools, same
deterministic outputs, but now the assistant can pick the right one
mid-conversation instead of you typing them yourself. The CLI surface
and on-disk project layout are unchanged; existing 0.3.1 users upgrade
in place.

### Added — Faz 6 (MCP server)
- New module `mlcompass.mcp_server` registering eight tools with a
  FastMCP server: `mlcompass_init`, `mlcompass_status`,
  `mlcompass_advise`, `mlcompass_audit`, `mlcompass_watch`,
  `mlcompass_compare`, `mlcompass_evaluate`, `mlcompass_deploy`. Tool
  names are prefixed so they don't collide with other MCP servers in
  the same client.
- New optional dependency group `mlcompass[mcp]` pulling `mcp>=1.2.0`.
  Pure CLI installs are unaffected; the import is guarded so the base
  package still works without the MCP SDK.
- New console script `mlcompass-mcp` (entry point
  `mlcompass.mcp_server:main`) for stdio JSON-RPC against Claude
  Desktop / Cursor / Claude Code.
- README gains a "Use from Claude Desktop / Cursor (MCP)" section with
  copy-pasteable `claude_desktop_config.json` and `.cursor/mcp.json`
  snippets and a tool-purpose table.

### Design notes
- MCP tools are **deterministic only** — the calling LLM is the
  interpreter, so there is no `--llm` knob on the MCP surface. The
  CLI's `--llm` reasoning modes stay where they are.
- `watch --apply` (config mutation) is intentionally **not exposed**
  via MCP. MCP doesn't standardise a confirm channel; mutating
  operations stay on the CLI behind the existing permission prompt.
- NaN / ±Inf metric values are normalised to `null` in tool outputs
  for JSON safety. The `nan` detector still surfaces them as findings.
- 23 new tests covering tool registry, every tool's success and error
  envelopes, the leakage-smell warning over MCP, and a NaN-survival
  test. Full suite now at 384 passing.
- `pytest-asyncio` added to the dev dependency group (`asyncio_mode =
  "auto"`); ruff stays at 0 errors and mypy `--strict` stays clean
  across 32 source files.

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
