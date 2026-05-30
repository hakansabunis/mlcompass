# mlcompass

> An LLM agent that sits next to you through your whole ML pipeline —
> from data, through training, all the way to deployment.

[![PyPI](https://img.shields.io/pypi/v/mlcompass.svg)](https://pypi.org/project/mlcompass/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

🚧 **Alpha (v0.3.1)** — under active development. APIs may change before v1.0.

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

## What's in v0.3

Eight commands — every stage of the ML pipeline plus a status
inspector.

| Command    | When you run it                          | What you get                                                  | Status |
| ---------- | ---------------------------------------- | ------------------------------------------------------------- | :----: |
| `init`     | Starting a new project                   | A `.mlcompass/` folder that tracks decisions                  | ✅ v0.1 |
| `advise`   | You have a CSV, what now?                | Models to try, features to derive, pitfalls to avoid          | ✅ v0.1 |
| `audit`    | Before you press train                   | Static analysis of training script (seed, val, optimizer, …)  | ✅ v0.2 |
| `watch`    | While training runs                      | Plateau / overfit / NaN / divergence (plain log / TB / W&B)   | ✅ v0.2 |
| `compare`  | After several runs                       | Side-by-side config + final-metric diff with verdict          | ✅ v0.2 |
| `evaluate` | Training done                            | Metrics, threshold sweep, confusion matrix, leakage-smell     | ✅ v0.3 |
| `deploy`   | Going to production                      | Model + deps + target-specific checks + production checklist  | ✅ v0.3 |
| `status`   | Any time                                 | Project metadata, active state, command activity, decisions   | ✅ v0.3 |

Every command except `init` and `status` keeps a fully deterministic
default path and offers an opt-in `--llm` flag that adds a
Claude-driven interpretation step on top.

## Install

```bash
pip install mlcompass
export ANTHROPIC_API_KEY="sk-ant-..."   # only needed for --llm modes
```

Optional extras:

```bash
pip install "mlcompass[tensorboard]"    # adds tbparse for TB event files
```

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
