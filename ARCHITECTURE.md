# mlcompass — Architecture

This document describes the design of mlcompass, an LLM agent that
assists ML practitioners across the full training pipeline.

It is the canonical reference for contributors and for anyone curious
about how the pieces fit together. Code is not the goal here — design
decisions, conventions, and seams are.

---

## 1. Vision

**One tool, one project, every stage.**

You start with a CSV. You end with a deployed model. mlcompass sits
next to you the entire time, keeping context across commands.

```
data.csv          train.py             results.csv         production
   │                  │                     │                  │
   ▼                  ▼                     ▼                  ▼
┌──────┐         ┌─────────┐          ┌──────────┐        ┌────────┐
│advise│ ──────► │  audit  │ ───────► │ evaluate │ ─────► │ deploy │
│      │         │  watch  │          │          │        │        │
│      │         │ compare │          │          │        │        │
└──────┘         └─────────┘          └──────────┘        └────────┘
   │                  │                     │                  │
   └──────────────────┴────────── shared ───┴──────────────────┘
                          project context
                         (.mlcompass/)
```

Every command writes to and reads from a shared project context.
By the time you reach `deploy`, the tool already knows your dataset,
your model choices, your training history, and your evaluation results.

---

## 2. Project context (`.mlcompass/`)

Each mlcompass project lives in a directory containing a `.mlcompass/`
folder, similar in spirit to `.git/`.

### Directory layout

```
.mlcompass/
├── project.yaml          # static metadata
├── context.json          # accumulated knowledge across commands
├── datasets/             # registered datasets
│   └── <fingerprint>.json
├── runs/                 # training run history
│   └── <run-id>/
│       ├── config.yaml
│       ├── metrics.json
│       └── notes.md
├── advice.log            # history of advisor recommendations
└── cache/                # LLM prompt cache, tool result cache
    └── ...
```

### `project.yaml` (static, written at `init`)

```yaml
name: customer-churn-model
created: 2026-05-29T14:30:00Z
mlcompass_version: 0.0.1
default_model: claude-opus-4-7
```

### `context.json` (dynamic, updated by every command)

```json
{
  "project_type": "binary_classification",
  "target_column": "churn",
  "preferred_models": ["xgboost", "lightgbm"],
  "active_dataset": "datasets/abc123.json",
  "current_run": "runs/run-42",
  "decisions": [
    {
      "timestamp": "2026-05-29T15:00:00Z",
      "command": "advise",
      "summary": "Recommended XGBoost as baseline",
      "reasoning": "Tabular binary classification, 10K rows, mixed types"
    }
  ]
}
```

### Why this matters

The persistent context is what makes mlcompass more than a chat tool:

- `advise` learns the project type and target column from the data.
- `audit` reads the project type to know whether validation split
  should be stratified.
- `watch` reads recent recommendations to interpret training behavior
  ("the user chose XGBoost but is now training a neural net — why?").
- `evaluate` already knows what good performance looks like for the
  chosen model family.
- `deploy` knows the best run.

---

## 3. Agent hierarchy

mlcompass uses agentlite for its agent backbone. Each command
dispatches to a specific agent, and most commands use sub-agents
to delegate focused subtasks.

```
┌─────────────────────────────────────────────────────────┐
│              CLI dispatcher (cli.py)                     │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   ┌─────────┐      ┌──────────┐      ┌─────────┐
   │ advise  │      │  watch   │      │ deploy  │
   │  agent  │      │  agent   │      │  agent  │
   └─────────┘      └──────────┘      └─────────┘
        │                 │                 │
        ▼                 ▼                 ▼
   ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
   │ ModelAdvisor │  │MetricsWatcher│  │ProductionAdv.│
   │  (Opus sub)  │  │ (Haiku sub)  │  │ (Opus sub)   │
   └──────────────┘  └─────────────┘  └──────────────┘
                          │
                          ▼
                    ┌──────────────┐
                    │Diagnostician │
                    │ (Opus, called│
                    │  on anomaly) │
                    └──────────────┘
```

### Model selection rules

| Agent role           | Default model         | Why                                  |
| -------------------- | --------------------- | ------------------------------------ |
| Orchestrators        | `claude-opus-4-7`     | Reasoning + planning                 |
| Domain advisors      | `claude-opus-4-7`     | Deep ML knowledge needed             |
| Watchers / pollers   | `claude-haiku-4-5`    | Cheap, runs constantly               |
| Static analyzers     | `claude-haiku-4-5`    | Pattern matching, not deep reasoning |
| Diagnosticians       | `claude-opus-4-7`     | Rare but high-value calls            |

Users can override with `--model` flag per command.

---

## 4. Tool catalog

Tools are organized by domain, not by command. Many tools are shared.

### Shared tools (used by multiple commands)

| Tool                              | Permission   | Description                                  |
| --------------------------------- | ------------ | -------------------------------------------- |
| `read_project_context()`          | none         | Read `.mlcompass/context.json`               |
| `write_project_context(updates)`  | none         | Append to context (project-internal)         |
| `list_datasets()`                 | none         | List registered datasets                     |
| `list_runs()`                     | none         | List training run history                    |
| `register_dataset(path, meta)`    | none         | Add a dataset to project                     |
| `get_active_dataset()`            | none         | Current dataset                              |

### Dataset / advise tools

| Tool                              | Permission   | Description                                  |
| --------------------------------- | ------------ | -------------------------------------------- |
| `analyze_dataset(path)`           | none         | Pandas-based schema + stats + outliers       |
| `compute_correlations(path)`      | none         | Correlation matrix for numeric cols          |
| `detect_target_column(path)`      | none         | Heuristic target detection                   |
| `suggest_models(analysis)`        | none         | Calls ModelAdvisor sub-agent                 |
| `suggest_features(analysis)`      | none         | Calls ModelAdvisor sub-agent                 |
| `generate_baseline_notebook()`    | **confirm**  | Writes a baseline notebook to disk           |

### Audit / script tools (Faz 2)

| Tool                              | Permission   | Description                                  |
| --------------------------------- | ------------ | -------------------------------------------- |
| `parse_python_script(path)`       | none         | AST parse                                    |
| `check_seed_setting(ast)`         | none         | Is `torch.manual_seed` etc. called?          |
| `check_validation_split(ast)`     | none         | Is val split reasonable?                     |
| `check_optimizer_config(ast)`     | none         | Common optimizer pitfalls                    |
| `check_loss_stability(ast)`       | none         | log/exp/div by zero risks                    |
| `apply_code_patch(file, patch)`   | **confirm**  | Modify user code                             |

### Watch / metrics tools (Faz 2)

| Tool                              | Permission   | Description                                  |
| --------------------------------- | ------------ | -------------------------------------------- |
| `read_tensorboard_log(path)`      | none         | Parse TB event files                         |
| `read_wandb_run(run_id)`          | none         | W&B local cache reader                       |
| `read_text_log(path)`             | none         | Regex-based plain-text log parser            |
| `detect_plateau(metrics)`         | none         | Moving-average plateau detection             |
| `detect_overfitting(metrics)`     | none         | Train/val gap analysis                       |
| `detect_nan(metrics)`             | none         | NaN / divergence trigger                     |
| `edit_config(file, updates)`      | **confirm**  | Modify YAML/JSON config                      |
| `restart_training(config)`        | **confirm**  | Re-launch training process                   |

### Evaluate tools (Faz 3)

| Tool                              | Permission   | Description                                  |
| --------------------------------- | ------------ | -------------------------------------------- |
| `analyze_results(path)`           | none         | Metric summary                               |
| `compute_confusion_matrix(...)`   | none         | CM + class-wise stats                        |
| `find_hard_examples(...)`         | none         | Top-k worst predictions                      |
| `suggest_threshold(probabilities)`| none         | Optimal classification threshold             |

### Deploy tools (Faz 4)

| Tool                              | Permission   | Description                                  |
| --------------------------------- | ------------ | -------------------------------------------- |
| `check_model_size(path)`          | none         | Bytes + parameter count                      |
| `check_dependencies()`            | none         | Frozen requirements vs imports               |
| `estimate_inference_latency()`    | none         | Sample-based timing estimate                 |
| `convert_to_onnx(model)`          | **confirm**  | Format conversion                            |

---

## 5. Permission strategy

mlcompass follows three trust levels, mapped to agentlite's
permission system.

### Level 0 — read-only (no permission)

Tools that read user files or project context without modification.
These run silently.

Examples: `analyze_dataset`, `read_tensorboard_log`, `check_seed_setting`.

### Level 1 — project-internal writes (silent)

Tools that write only inside `.mlcompass/`. Since this directory
belongs to mlcompass, writes here don't disturb user code.

Examples: `write_project_context`, `register_dataset`, `save_run_metrics`.

### Level 2 — user file modifications (always confirm)

Tools that touch user code, config, or trigger user-facing actions.
**Always** prompt before executing.

Examples: `apply_code_patch`, `restart_training`, `edit_config`,
`generate_baseline_notebook`, `convert_to_onnx`.

### Confirmation UX

Permission prompts use agentlite's `confirm_fn` plugged into a rich
terminal prompt:

```
⚠  mlcompass wants to edit train.py:

   Add: torch.manual_seed(42)
   Line: 12 (top of training function)

   Reason: reproducibility — seed is currently unset

   Apply? [y/N/diff]
```

The `diff` option shows the full proposed change before deciding.

### Why this matters

Most agentic ML tools either:
- (a) act autonomously and break things (low trust), or
- (b) only suggest and never act (low value).

mlcompass uses agentlite's first-class permission model to land in
between: **the agent proposes, the human approves, the agent executes.**

---

## 6. Module organization

```
src/mlcompass/
├── __init__.py              # __version__
├── cli.py                   # Click-based command dispatcher
├── context.py               # ProjectContext class
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py      # Common base / factory
│   ├── advise.py
│   ├── audit.py             # Faz 2
│   ├── watch.py             # Faz 2
│   ├── evaluate.py          # Faz 3
│   └── deploy.py            # Faz 4
├── tools/
│   ├── __init__.py
│   ├── context_tools.py     # project context @tool wrappers
│   ├── dataset.py           # analyze_dataset etc.
│   ├── script.py            # Faz 2: AST parsing
│   ├── logs.py              # Faz 2: log parsers
│   ├── metrics.py           # Faz 2: anomaly detection
│   ├── eval.py              # Faz 3
│   └── deploy.py            # Faz 4
├── ui/
│   ├── __init__.py
│   ├── advise.py            # rich rendering for advise output
│   ├── audit.py             # Faz 2
│   ├── watch.py             # Faz 2
│   └── common.py            # shared rich helpers (banners, tables)
├── prompts/
│   ├── orchestrator.txt
│   ├── advisor.txt
│   ├── auditor.txt          # Faz 2
│   ├── diagnostician.txt    # Faz 2
│   ├── results_analyzer.txt # Faz 3
│   └── production.txt       # Faz 4
└── version.py               # Single source of __version__
```

### Conventions

- **One agent per file.** Each agent in `agents/` is a single Python
  module exporting a factory function `build_<name>_agent(...)`.
- **System prompts live in `prompts/`** as plain text (not inlined in
  Python). This lets non-developers contribute prompt tweaks via PR
  without touching code.
- **Tools are grouped by domain**, not by command. `analyze_dataset`
  lives in `tools/dataset.py` regardless of which command uses it.
- **Rich rendering separated** from agent logic. The `ui/` layer
  takes agent output structures and renders them to terminal.

---

## 7. CLI structure

```
mlcompass <command> [args] [options]

Commands:
  init      <name>          Initialize a new project
  advise    <data>          Analyze data, recommend models + features
  audit     <script>        Static analysis of training script  [Faz 2]
  watch     <script>        Live training monitor               [Faz 2]
  compare   <r1> <r2>       Compare two training runs           [Faz 2]
  evaluate  <results>       Post-training analysis              [Faz 3]
  deploy    --target <X>    Deployment check + advice           [Faz 4]
  status                    Show current project context

Common options:
  --project DIR             .mlcompass path (default: ./.mlcompass)
  --model NAME              LLM model override
  --no-color                Disable rich UI
  --verbose, -v             Show agent reasoning step-by-step
  --yes, -y                 Auto-approve all confirmations (dangerous)
  --help, -h                Show help for command
```

### Example sessions

```bash
# Day 1 — new project, exploring data
$ mlcompass init churn-model
✓ Created .mlcompass/

$ mlcompass advise data/customers.csv
[shows dataset analysis + model recommendations]

# Day 2 — running first training
$ mlcompass audit train.py
[shows static issues found]

$ mlcompass watch train.py
[live monitoring begins]

# Day 5 — checking results
$ mlcompass evaluate runs/run-42/predictions.csv
[shows post-training analysis]

$ mlcompass compare run-3 run-42
[shows hypothesis-driven diff]

# Day 7 — going to production
$ mlcompass deploy --target sagemaker
[deployment checklist]
```

---

## 8. Data flow between phases

```
┌─────────┐     init      ┌──────────────────┐
│  user   │  ─────────►   │  project.yaml    │
└─────────┘                │  context.json    │
                           └──────────────────┘
                                   │
                                   ▼
┌─────────┐     advise    ┌──────────────────┐
│  user   │  ─────────►   │  + active_dataset│
│  + CSV  │               │  + recommendations│
└─────────┘                │  + decisions[]   │
                           └──────────────────┘
                                   │
                                   ▼
┌─────────┐     watch     ┌──────────────────┐
│  user   │  ─────────►   │  + current_run   │
│ +train  │               │  + runs/<id>/    │
└─────────┘                │     metrics.json │
                           │     config.yaml  │
                           └──────────────────┘
                                   │
                                   ▼
┌─────────┐    evaluate   ┌──────────────────┐
│  user   │  ─────────►   │  + runs/<id>/    │
│+results │               │     evaluation   │
└─────────┘                │     .json        │
                           └──────────────────┘
                                   │
                                   ▼
┌─────────┐     deploy    ┌──────────────────┐
│  user   │  ─────────►   │  deployment      │
│         │               │  report (rendered│
│         │               │  to terminal)    │
└─────────┘                └──────────────────┘
```

The arrow direction is one-way (downstream). Earlier commands never
read state written by later commands. This keeps the system simple
and avoids circular reasoning.

---

## 9. Extension points

For future contributors, the following seams are designed to be
extended:

### Adding a new mode

1. Add a new file under `agents/` exporting `build_<name>_agent(...)`.
2. Add a system prompt under `prompts/<name>.txt`.
3. Add corresponding rendering under `ui/<name>.py`.
4. Register the command in `cli.py`.

### Adding a new tool

1. Choose the appropriate file under `tools/` based on domain.
2. Decorate with `@tool` from agentlite.
3. Mark `requires_confirmation=True` if the tool modifies user state.
4. Import in the relevant agent file.

### Adding a new log format (Faz 2)

1. Add a parser function to `tools/logs.py`.
2. Make it return the standard `MetricSnapshot` structure (see
   `tools/metrics.py`).
3. Register a file-extension detector.

### Adding a new deployment target (Faz 4)

1. Add a checker function to `tools/deploy.py`.
2. Register under the `--target` option in `cli.py`.

---

## 10. Open questions (to resolve before Faz 1 implementation)

| Question                                              | Default answer                          |
| ----------------------------------------------------- | --------------------------------------- |
| Should context be JSON or SQLite?                     | JSON (Faz 1), revisit at v0.5           |
| How do we handle parallel runs (e.g., grid search)?   | Faz 2 problem, defer                    |
| Should `advise` fetch real benchmark data online?     | No (Faz 1), maybe Faz 3                 |
| Multi-language project support (R, Julia)?            | Python-first; revisit at v1.0           |
| Hosted version (SaaS) or local-only?                  | Local-only at v0.x; SaaS at v1.0+       |
| How does it handle very large datasets (>10GB CSV)?   | Sample first 100K rows for advise       |
| Should we ship a default `.mlcompass/.gitignore`?     | Yes — include `cache/` and `runs/`      |

---

## 11. Non-goals

To stay focused, mlcompass will **not** try to be:

- **An AutoML system.** We advise; we don't auto-train hundreds of
  models. Use AutoSklearn, AutoGluon, etc. for that.
- **A model registry or experiment tracker.** Use MLflow or W&B.
  mlcompass's `runs/` is for context, not auditing.
- **A general code assistant.** Use Cursor, Copilot, or aider.
- **A data labeling tool.** Use Label Studio, Snorkel, etc.
- **A monitoring dashboard.** Use Grafana, Streamlit, etc.

mlcompass is the **advisor that sits next to all of these tools**,
not a replacement for any of them.

---

## 12. Glossary

- **Project context** — the persistent state stored in `.mlcompass/`
- **Run** — a single training execution with associated metrics
- **Recommendation** — an advisor sub-agent's structured suggestion
- **Permission gate** — agentlite's `requires_confirmation=True` hook
- **Sub-agent** — an agentlite agent spawned via `subagent(...)`
- **Snapshot** — a single point-in-time metric reading (epoch, step)

---

_Last updated: 2026-05-29 — covers Faz 1 design, Faz 2–4 sketched._
