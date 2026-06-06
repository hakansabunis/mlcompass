# mlcompass

> An LLM agent that sits next to you through your whole ML pipeline —
> from data, through training, all the way to deployment.

[![PyPI](https://img.shields.io/pypi/v/mlcompass.svg)](https://pypi.org/project/mlcompass/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

🚧 **Alpha (v0.8.0)** — under active development. APIs may change before v1.0.

📄 **Paper:** mlcompass is described in [our paper](Capstone_Report.pdf) — *"mlcompass: A Schema-Bounded LLM Narrator for Machine-Learning Pipeline Diagnosis"*. The paper's anti-hallucination ablation (Table I) is reproducible end-to-end via the scripts in [`scripts/`](scripts/). See [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) for what the contract does and does not protect against, and [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) for what we have not yet evaluated.

## What it does

mlcompass is a single CLI that follows your ML project from data to
production, keeping context across every step.

```
data.csv         train.py          two runs        results.csv      production
   │                │                  │                │                │
   ▼                ▼                  ▼                ▼                ▼
 advise   ────►   audit   ────►    compare     ────► evaluate ─────►  deploy
                  watch
```

Each command writes to and reads from a shared project context
(`.mlcompass/`), so by the time you reach `deploy`, the tool already
knows your dataset, your model choice, your training history, and your
evaluation results.

## What's in v0.8

Eleven commands cover every stage of the ML pipeline. **v0.8 ships
eleven ready-made Claude Code slash commands** so the same eleven
tools become one-keystroke calls inside Claude Code — one
`mlcompass install-slash-commands` and you get `/mlc-advise`,
`/mlc-evaluate`, `/mlc-leak`, … as first-class slash entries.

v0.7 introduced **automatic leakage investigation** with an
anti-hallucination contract — when `evaluate` sees a suspiciously
perfect metric, it gathers structured evidence about which columns
might be the source and (with `--llm`) hands it to a Claude agent
that's forbidden from inventing column names or proposing code
patches.

| Command    | When you run it                          | What you get                                                  | Status |
| ---------- | ---------------------------------------- | ------------------------------------------------------------- | :----: |
| `init`     | Starting a new project                   | A `.mlcompass/` folder that tracks decisions                  | ✅ v0.1 |
| `advise`   | You have a CSV, what now?                | Models to try, features to derive, pitfalls to avoid          | ✅ v0.1 |
| `audit`    | Before you press train                   | Static analysis of training script (seed, val, optimizer, …)  | ✅ v0.2 |
| `watch`    | While training runs                      | Plateau / overfit / NaN / divergence (plain log / TB / W&B)   | ✅ v0.2 |
| `compare`  | After several runs                       | Side-by-side config + final-metric diff with verdict          | ✅ v0.2 |
| `evaluate` | Training done                            | Metrics, threshold sweep, confusion matrix, **leakage investigation** | ✅ v0.3 / **v0.7 panel** |
| `deploy`   | Going to production                      | Model + deps + target-specific checks + production checklist  | ✅ v0.3 |
| `status`   | Any time                                 | Project metadata, active state, command activity, decisions   | ✅ v0.3 |
| `agent`    | "Just do it for me"                      | LLM-driven router across the other tools, with memory         | ✅ v0.5 |
| `monitor`  | Model deployed; new data flowing         | PSI + KS + chi² drift across features, retrain verdict        | ✅ v0.6 |
| `optimize` | You have a few runs; what's next?        | HPO sub-agent: leaderboard, sensitivity, N suggested configs  | ✅ v0.6 |

Every command except `init`, `status`, and `agent` keeps a fully
deterministic default path and offers an opt-in `--llm` flag that adds
a Claude-driven interpretation step on top. The `agent` command is
the inverse: LLM-first by design, with the other tools as its hands —
and now remembers across runs via per-project memory.

**Battle-tested**: v0.6 → v0.7.3 ships **4 field-test patches** that
closed bugs surfaced by real Kaggle datasets (Telco Churn, Ames House
Prices, Titanic, Penguins, Insurance Charges). Each release went
through a live dry run before shipping — see
[CHANGELOG.md](CHANGELOG.md) for the full log.

## Install

```bash
pip install mlcompass
export ANTHROPIC_API_KEY="sk-ant-..."   # only needed for --llm modes
```

Optional extras:

```bash
pip install "mlcompass[tensorboard]"          # adds tbparse for TB event files
pip install "mlcompass[mcp]"                  # adds the Claude / Cursor MCP server
pip install "mlcompass[agent]"                # adds the self-driving agent (anthropic API)
pip install "mlcompass[agent-claude-code]"    # alt agent backend via Claude Code CLI
```

## Use from Claude Desktop / Cursor (MCP)

mlcompass ships a **Model Context Protocol** server, so any MCP-capable
client (Claude Desktop, Claude Code, Cursor, Continue, …) can call its
eight tools directly — you describe the situation in natural language,
the assistant picks the right `mlcompass_*` tool and feeds the result
back into the conversation.

```bash
pip install "mlcompass[mcp]"
```

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`
on macOS, `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "mlcompass": {
      "command": "mlcompass-mcp"
    }
  }
}
```

**Cursor** (`.cursor/mcp.json` in your project, or `~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "mlcompass": {
      "command": "mlcompass-mcp"
    }
  }
}
```

**Claude Code** (preferred: a project-local `.mcp.json` so the
server only loads in the project that needs it):

```json
{
  "mcpServers": {
    "mlcompass": {
      "command": "C:\\path\\to\\your\\project\\.venv\\Scripts\\mlcompass-mcp.exe"
    }
  }
}
```

On the first run Claude Code asks whether to trust the MCP server —
pick "Use this MCP server" (the narrowest option). Verify with the
`/mcp` slash command; you should see `mlcompass` listed with all
eight tools available.

### Use as Claude Code slash commands (v0.8)

`pip install mlcompass` ships **eleven ready-made Claude Code slash
commands**. One command installs them into Claude Code's commands
directory:

```bash
mlcompass install-slash-commands           # → .claude/commands/ in cwd
mlcompass install-slash-commands --scope user   # → ~/.claude/commands/
mlcompass install-slash-commands --force        # overwrite local edits
```

Restart Claude Code (or reload the project) and type the slash key —
the eleven commands appear in the slash menu, each one prefixed
`mlc-` so they don't collide with the underlying MCP tool names
(Claude Code routes `/advise` to the MCP tool browser; `/mlc-advise`
fires our command):

| Slash command         | When you type it…                                         |
| --------------------- | --------------------------------------------------------- |
| `/mlc-init <name>`    | Start a new mlcompass project here                        |
| `/mlc-advise <csv>`   | Look at this dataset and tell me what to do               |
| `/mlc-audit <script>` | Review my training script before I press train            |
| `/mlc-watch <log>`    | Look at this training log and flag anything weird         |
| `/mlc-compare <a> <b>`| Which of these two runs is better, and why?               |
| `/mlc-evaluate <csv>` | Read these predictions and tell me what they mean         |
| `/mlc-leak <csv>`     | Focused leakage check — anti-hallucination panel only     |
| `/mlc-deploy <model>` | Is this model ready to ship?                              |
| `/mlc-status`         | What does this project look like right now?               |
| `/mlc-monitor <ref> <cur>` | Drift check against a reference dataset (CLI fallback) |
| `/mlc-optimize <metric>`   | What hyperparameters to try next? (CLI fallback)       |

`/mlc-evaluate` and `/mlc-leak` honour the same anti-hallucination
contract as the CLI — if the leakage investigation is absent,
Claude won't invent column names; if present, it can cite **only**
items in the evidence panel.

`/mlc-monitor` and `/mlc-optimize` shell out to the CLI rather than
the MCP server (these two tools aren't exposed via MCP in v0.7.x);
everything else runs through the `mlcompass-mcp` tool layer.

The install command is idempotent: a second run skips files you've
already customised. Add `--force` if you've upgraded mlcompass and
want the new versions.

Restart the client and the eight tools appear:

| Tool                  | Use it when…                                            |
| --------------------- | ------------------------------------------------------- |
| `mlcompass_init`      | Starting a new project                                  |
| `mlcompass_advise`    | Asking the assistant to look at a dataset               |
| `mlcompass_audit`     | Asking the assistant to review a training script        |
| `mlcompass_watch`     | Pointing the assistant at a training log / TB / W&B run |
| `mlcompass_compare`   | "Which of these two runs is better, and why?"           |
| `mlcompass_evaluate`  | "Read these predictions and tell me what they mean"     |
| `mlcompass_deploy`    | "Is this model ready to ship to Lambda?"                |
| `mlcompass_status`    | "What does this project look like right now?"           |

All tools are **deterministic** — the assistant reads their structured
output and does its own interpretation, with full access to your
conversation's context. The CLI stays available for scripted use and
for the `--llm` reasoning modes.

## Use as a self-driving agent (CLI)

When you're not in Claude Desktop — CI runs, cron jobs, an ssh session
on a GPU box — you can let an agent drive the same eight tools from
the terminal:

```bash
pip install "mlcompass[agent]"
export ANTHROPIC_API_KEY="sk-ant-..."

mlcompass agent "I have data.csv, take me from raw data to a model recommendation"
```

The agent picks tools (`mlcompass_advise`, then `mlcompass_status`,
then …), streams every reasoning step + tool call + tool result to the
terminal, and writes a transcript under
`.mlcompass/agent_runs/<id>/transcript.jsonl` plus a human-readable
`summary.md` next to it.

### Two backends

| Backend            | Dependency              | Best for                                    |
| ------------------ | ----------------------- | ------------------------------------------- |
| `api` *(default)*  | `mlcompass[agent]`      | Universal: API key + nothing else.          |
| `claude-code`      | `mlcompass[agent-claude-code]` + the `claude` CLI on PATH | Power users already on Claude Code; routes through Anthropic's official Agent SDK. |

```bash
# Default — talks straight to the Anthropic API.
mlcompass agent "Compare run-3 and run-7" --project-path .

# Alt — routes through your local Claude Code CLI.
pip install "mlcompass[agent-claude-code]"
mlcompass agent "Audit train.py and tell me what to fix" --backend claude-code

# Headless / CI — skip the y/N permission prompt for mutating tools.
mlcompass agent "Init a new churn project here" --auto-approve

# Cap the safety budget if you're worried about runaway loops.
mlcompass agent "Diagnose this run" --max-turns 10 --model claude-sonnet-4-5
```

The agent will **ask before mutating** by default — the only mutating
tool is `mlcompass_init`. Read/compute tools (`advise`, `audit`,
`watch`, `compare`, `evaluate`, `deploy`, `status`) auto-allow. Add
`--auto-approve` to skip the prompt for headless runs.

## Five-minute tour

```bash
mlcompass init my-project

# Pre-training
mlcompass advise data.csv --target churn

# Training-time
mlcompass audit train.py                     # static checks
mlcompass audit train.py --llm               # + prioritized synthesis
mlcompass watch train.log                    # one-shot plain-text scan
mlcompass watch runs/tb_run/                 # TensorBoard event files
mlcompass watch wandb/run-001/               # W&B local run directory
mlcompass watch train.log --follow           # live tail (plain-text only)
mlcompass watch train.log --llm              # + diagnostician
mlcompass watch train.log --llm \            # + permission-gated edits
  --apply --config train.yaml                #   (prompted per change)

# Comparing runs
mlcompass compare run-3 run-7                # deterministic diff
mlcompass compare run-3 run-7 --llm          # + hypothesis + next experiment

# Post-training
mlcompass evaluate results.csv               # metrics + threshold sweep
mlcompass evaluate results.csv --llm         # + assessment + next steps

# Deployment
mlcompass deploy model.pt                    # model + checklist
mlcompass deploy model.pt --requirements reqs.txt --target lambda
mlcompass deploy model.pt --llm              # + production verdict

# Any time — what's the project look like right now?
mlcompass status
mlcompass status --recent 10                 # last 10 decisions

# Let the agent drive the whole pipeline
mlcompass agent "I have data.csv, take me to a deployed model"
mlcompass agent "Compare run-3 and run-7" --backend claude-code
mlcompass agent "Init a new project here" --auto-approve
mlcompass agent "Continue what we started" --resume 20260530-150000

# Post-deploy drift check
mlcompass monitor reference.csv current.csv             # PSI/KS/chi² per feature
mlcompass monitor reference.csv current.csv --llm       # + LLM interpretation

# What hyperparameters to try next?
mlcompass optimize --metric val_acc                     # reads .mlcompass/runs/
mlcompass optimize --metric val_acc --llm               # + strategist plan
mlcompass optimize --metric val_acc \
  --constraints "lr:0.0001-0.1,batch_size:16-256"       # bound the search
```

## Example — `advise`

```bash
mlcompass advise examples/customer_churn.csv
```

```
📊 Dataset analysis
   Path:    examples/customer_churn.csv
   Shape:   500 rows × 8 columns
   Target:  churn (high confidence)
   Task:    binary classification (0=98%, 1=2%)

⚠ Warnings
  • Class imbalance detected (1.6% minority class). Don't optimise
    accuracy — use AUC/F1/recall@k. Consider class_weight='balanced'
    or focal loss.

✨ Recommended models  (with --llm)
  • XGBoost                 AUC 0.78 – 0.83
  • Logistic Regression     AUC 0.70 – 0.74
  • LightGBM                AUC 0.78 – 0.84
```

## Example — `audit`

```bash
mlcompass audit train.py
```

```
🔎 Script audit
   Path: train.py | Lines: 23 | Frameworks: torch

   ✗ error    seed              No random seed set anywhere
   ✗ error    optimizer   L17   Adam does not accept momentum=
   ⚠ warning  val_split         No validation split detected
   ⚠ warning  grad_clipping L8  LSTM but no clip_grad_norm_
   ⚠ warning  dataloader  L20   DataLoader missing shuffle=
   ⚠ warning  loss_stability L23 log(x) without epsilon clipping
   ℹ info     batch_size  L20   batch_size=1 is very small

   Summary: 2 error   4 warning   1 info
```

Eight pure-AST rules:

| Rule              | Catches                                                       |
| ----------------- | ------------------------------------------------------------- |
| `seed`            | No `torch.manual_seed` / `np.random.seed` / `set_seed` call  |
| `val_split`       | No split detected, or split implausibly small                 |
| `optimizer`       | Adam-family + `momentum=`, weird lr, SGD without momentum     |
| `loss_stability`  | `log(x)` / `np.log(x)` without clamp or epsilon               |
| `dataloader`      | `DataLoader(...)` without explicit `shuffle=`                 |
| `grad_clipping`   | RNN / Transformer built but `clip_grad_norm_` never called    |
| `eval_mode`       | `model.train()` appears but `.eval()` never does              |
| `batch_size`      | Implausibly small (<4) or huge (>4096)                        |

## Example — `watch`

```bash
mlcompass watch train.log
```

```
👁  Watch report
   Log:        train.log
   Snapshots:  9
   Last epoch: 7
   Findings:   1 warning

Recent metrics (last 8)
┌───────┬────────────┬──────────┬─────────┐
│ Epoch │ train_loss │ val_loss │ val_acc │
├───────┼────────────┼──────────┼─────────┤
│   0   │       0.65 │     0.68 │   0.612 │
│   …   │        …   │      …   │    …    │
│   7   │       0.08 │     0.59 │   0.773 │
└───────┴────────────┴──────────┴─────────┘

⚠ warning  overfitting  L7  train_loss dropped -0.17 but val_loss
                            rose +0.11; current gap is 0.51
```

Four detectors:

| Rule           | Triggers when                                              |
| -------------- | ---------------------------------------------------------- |
| `nan`          | Any loss-like metric becomes NaN or ±Inf                   |
| `divergence`   | Train loss jumps ≥10× between consecutive snapshots        |
| `plateau`      | Primary loss flat across the last 5 snapshots              |
| `overfitting`  | Train falling, val rising, with a meaningful gap           |

Add `--follow` to tail the log file and surface new findings live.

## Example — `compare`

```bash
mlcompass compare run-3 run-7
```

```
🆚 Run comparison
   Run A  run-3  (baseline)             · 20 epochs
   Run B  run-7  (lower-lr-more-dropout) · 20 epochs

Final-epoch metrics
   Metric      Run A    Run B    Δ (B − A)   Winner
   train_loss  0.18     0.24     +0.06       A
   val_acc     0.79     0.87     +0.08       B
   val_loss    0.42     0.28     -0.14       B

Config differences
   dropout     0.1      0.3
   lr          0.001    0.0003

⚖️ Mixed result: A wins 1, B wins 2, 0 tie(s).
```

## Example — `evaluate` with automatic leakage investigation (v0.7)

When `evaluate` sees a metric that looks too good to be true (AUC >
0.995, accuracy > 0.99, or R² > 0.999), it doesn't just warn — it
runs a **deterministic investigator** that gathers structured
evidence about which columns might be the source:

```bash
mlcompass evaluate predictions.csv
```

```
⚠ Warnings
  • Suspiciously high R² (1.0000). On real-world data this almost always
    means data leakage (target value present in features), train/test
    contamination, or that the predictions table was scored on the
    training set.

┌──── 🔬 Leakage investigation — evidence ────┐
│ Suspicious metric: r2 = 1.0                 │
│ Candidate leak columns: log_price           │
│ y_pred == y_true match rate: 0.9%           │
│                                             │
│ Top correlations with target:               │
│   • log_price        r=+1.0000 (spearman)   │
│   • overall_qual     r=+0.8088 (spearman)   │
│   • sqft             r=+0.7233 (spearman)   │
└─────────────────────────────────────────────┘
```

The panel is **always free of charge** — no flag needed, no LLM
involved. With `--llm`, a second panel narrates the evidence under a
strict **anti-hallucination contract**: the agent may only cite items
present in the evidence dict, must say `cannot_determine` rather than
speculate, and is forbidden from proposing code patches (only manual
checks). **Facts → narration**, not the other way around.

What gets caught:
- **Exact label copy** — a feature whose Pearson correlation with
  the target is ≥ 0.99.
- **Monotone leaks** (`log(target)`, `sqrt(target)`, …) — Pearson sits
  at ~0.94 but Spearman = 1.0; the detector reports
  `max(|Pearson|, |Spearman|)` so the leak doesn't slip past the
  threshold.
- **Perfect-match predictions** — when `y_pred == y_true` ≥ 95% on a
  non-trivial task, that's a smoking gun for train/test contamination.

## Reproducing the paper's anti-hallucination ablation

The paper's Table I (phantom-column fabrication rate under three
contract configurations) is reproducible end-to-end via two scripts
under [`scripts/`](scripts/). The default mode runs deterministically
and requires no API key:

```bash
# Reproduces Table I in mock mode (~5 seconds, no API calls).
python scripts/reproduce_hallucination_ablation.py

# Tight confidence intervals at N=2000 (~$48 in API charges).
python scripts/reproduce_hallucination_ablation.py --mode live --n 2000

# Measure end-to-end latency and Layer-3 retry rate.
python scripts/measure_latency.py --live --n 50
```

Output is a markdown table with Wilson 95% confidence intervals
matching the format used in the paper. Pass `--json` for a
machine-readable summary suitable for CI consumption.

See [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) for what the
contract protects against and what it does not.
See [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) for the
limitations that the v0.8.0 release has not yet addressed.
See [`docs/FIELD_TEST_BUGS.md`](docs/FIELD_TEST_BUGS.md) for per-bug
detail on the eleven field-test findings categorized in Table II.

## Why mlcompass

The ML ecosystem already has great tools — but each owns one slice of
the pipeline, and none of them advise:

|                                 | pandas-profiling | W&B / TensorBoard | Cursor / Devin | **mlcompass** |
| ------------------------------- | :--------------: | :---------------: | :------------: | :------------: |
| Analyzes raw data               |        ✅        |         ❌        |       ❌       |       ✅       |
| Recommends models + features    |        ❌        |         ❌        |     partial    |       ✅       |
| Audits training scripts         |        ❌        |         ❌        |     reactive   |       ✅       |
| Watches training in real time   |        ❌        |    dashboard      |       ❌       |       ✅       |
| Diagnoses problems proactively  |        ❌        |         ❌        |     reactive   |       ✅       |
| Persistent project memory       |        ❌        |    per-run        |       ❌       |       ✅       |
| Permission-gated actions        |        ❌        |         ❌        |     partial    |   first-class  |

mlcompass is the **advisor that sits next to all of these tools** —
not a replacement for any.

## How it works

Built on [agentlite](https://github.com/hakansabunis/agentlite) — a
small Claude agent library — mlcompass uses one deterministic analyzer
per command (pure pandas / pure AST / pure log parser) plus an optional
LLM agent layer that runs on top of the analyzer's structured output.

```
        cli.py
          │
   ┌──────┼──────┬─────────┬──────────┐
   ▼      ▼      ▼         ▼          ▼
 init  advise  audit     watch     compare
                │         │           │
                ▼         ▼           ▼
            (--llm)    (--llm)     (--llm)
            priori-   diagnos-   hypothes-
            tizer     tician     izer
```

Every action that would modify your code, config, or run a training
process **asks permission first** — agentlite's permission system is
first-class, not an afterthought.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Project context

Each mlcompass project keeps a small folder, similar in spirit to
`.git/`:

```
.mlcompass/
├── project.yaml        # metadata
├── context.json        # decisions, recommendations, active state
├── datasets/           # registered datasets
├── runs/               # training run history (consumed by compare)
└── advice.log          # JSONL of every command run
```

This is what makes mlcompass more than a chat tool: by the time you
run `deploy`, every earlier decision is still in memory.

## Roadmap

| Phase                | Commands                              |  Status        |
| -------------------- | ------------------------------------- | :------------: |
| **Faz 1 (v0.1)**     | `init`, `advise`                      | ✅ Shipped      |
| **Faz 2 (v0.2)**     | `audit`, `watch`, `compare` + `--llm` | ✅ Shipped      |
| **Faz 2.2 (v0.3)**   | TensorBoard / W&B sources, `--apply`  | ✅ Shipped      |
| **Faz 3 (v0.3)**     | `evaluate` + leakage-smell warning    | ✅ Shipped      |
| **Faz 4 (v0.3)**     | `deploy`                              | ✅ Shipped      |
| **Faz 5 (v0.3)**     | `status`                              | ✅ Shipped      |
| **Faz 6 (v0.4)**     | MCP server — `mlcompass-mcp`          | ✅ Shipped      |
| **Faz 7 (v0.5)**     | `agent` — self-driving (api + claude-code backends) | ✅ Shipped |
| **Faz 8 (v0.6)**     | `monitor` + `optimize` + agent memory   | ✅ Shipped      |
| **Faz 9 (v0.7)**     | Automatic leakage investigation + anti-hallucination contract | ✅ Shipped |
| **v0.7.1 — v0.7.3**  | Field-test patches (Titanic + Penguins + Insurance) | ✅ Shipped |
| **Faz 10 (v0.8)**    | Claude Code slash commands — `mlcompass install-slash-commands` + 11 `/mlc-*` entries | ✅ Shipped |

See [CHANGELOG.md](CHANGELOG.md) for the detailed log and
[ARCHITECTURE.md](ARCHITECTURE.md) for the design.

## Non-goals

To stay focused, mlcompass will **not** try to be:

- **AutoML** (use AutoGluon, AutoSklearn)
- **Experiment tracker** (use MLflow, W&B)
- **Code assistant** (use Cursor, Copilot, aider)
- **Monitoring dashboard** (use Grafana, Streamlit)

mlcompass **advises**; you decide.

## Contributing

Alpha-stage — issues and discussions welcome, see
[CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup.

## License

MIT © 2026 Hakan Sabunis
