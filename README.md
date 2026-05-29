# ml-copilot

> An LLM agent that sits next to you through your whole ML pipeline —
> from data, through training, all the way to deployment.

[![PyPI](https://img.shields.io/pypi/v/ml-copilot.svg)](https://pypi.org/project/ml-copilot/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

🚧 **Pre-alpha (v0.0.1)** — under active development. APIs will change before v0.1.

## What it does

ml-copilot is a single CLI that follows your ML project from start
to finish, keeping context across every step.

```
data.csv          train.py             results.csv         production
   │                  │                     │                  │
   ▼                  ▼                     ▼                  ▼
 advise   ────►   audit + watch  ────►  evaluate  ────►  deploy
                      compare
```

Each command writes to and reads from a shared project context
(`.mlcopilot/`), so by the time you reach `deploy`, the tool already
knows your dataset, your model choice, your training history, and
your evaluation results.

## Six commands, one tool

| Command    | When you run it                          | What you get                                          |
| ---------- | ---------------------------------------- | ----------------------------------------------------- |
| `init`     | Starting a new project                   | A `.mlcopilot/` folder that tracks decisions          |
| `advise`   | You have a CSV, what now?                | Models to try, features to derive, pitfalls to avoid  |
| `audit`    | Before you press train                   | Static analysis of training script (seed, val, etc.)  |
| `watch`    | While training runs                      | Live plateau / overfit / NaN detection                |
| `compare`  | After several runs                       | Hypothesis-driven diff between two runs               |
| `evaluate` | Training done                            | Threshold tuning, confusion matrix, hard examples     |
| `deploy`   | Going to production                      | Latency estimate, dependency check, ONNX advice       |

## Quick example — `advise` mode

```bash
ml-copilot init churn-project
ml-copilot advise data/customers.csv --target churn
```

Output:

```
📊 Dataset analysis (data/customers.csv)
   • 10,000 rows × 23 columns
   • Target: churn (binary, 12% positive)
   • 4 categorical, 18 numerical, 1 datetime
   • 3 columns with >50% missing values (consider dropping)

💡 Recommended models
   1. XGBoost / LightGBM   → tabular binary baseline
                             expected AUC: 0.82 – 0.87
   2. Logistic Regression  → interpretable baseline
                             expected AUC: 0.76 – 0.80
   3. FT-Transformer       → if GPU budget allows
                             expected AUC: 0.83 – 0.86

🔧 Suggested feature engineering
   • signup_date → derive days_since_signup, month, dayofweek
   • income (3 outliers >3σ) → winsorize at 99th percentile
   • country (47 categories) → target encoding or top-N

⚠️  Class imbalance (12% positive)
   • Don't optimize accuracy — use AUC, F1, or recall@k
   • Consider class_weight='balanced' or focal loss

Generate a baseline notebook? [y/N]
```

## Quick example — `watch` mode (Faz 2)

```bash
ml-copilot watch train.py
```

After 8 epochs:

```
⚠️  Epoch 8 — overfitting detected
   Train loss: 0.118  |  Val loss: 0.387  (gap 0.27, normal <0.1)

   Likely cause: regularization is too weak for the model capacity.

   Suggested fix: increase dropout 0.1 → 0.3
   Apply and restart training? [y/N]
```

## Why ml-copilot

The ML ecosystem already has great tools — but each owns one slice
of the pipeline, and none of them advise:

|                                 | pandas-profiling | W&B / TensorBoard | Cursor / Devin | **ml-copilot** |
| ------------------------------- | :--------------: | :---------------: | :------------: | :------------: |
| Analyzes raw data               |        ✅        |         ❌        |       ❌       |       ✅       |
| Recommends models + features    |        ❌        |         ❌        |     partial    |       ✅       |
| Audits training scripts         |        ❌        |         ❌        |     reactive   |       ✅       |
| Watches training in real time   |        ❌        |    dashboard      |       ❌       |       ✅       |
| Diagnoses problems proactively  |        ❌        |         ❌        |     reactive   |       ✅       |
| Post-training evaluation advice |        ❌        |       basic       |       ❌       |       ✅       |
| Deployment readiness check      |        ❌        |         ❌        |       ❌       |       ✅       |
| Persistent project memory       |        ❌        |    per-run        |       ❌       |       ✅       |
| Permission-gated actions        |        ❌        |         ❌        |     partial    |   first-class  |

ml-copilot is the **advisor that sits next to all of these tools** —
not a replacement for any.

## Install

```bash
pip install ml-copilot
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

```bash
# Start a project
ml-copilot init my-project

# Pre-training
ml-copilot advise data.csv --target label

# Training-time          (Faz 2)
ml-copilot audit train.py
ml-copilot watch train.py
ml-copilot compare run-3 run-7

# Post-training          (Faz 3)
ml-copilot evaluate results.csv

# Deployment             (Faz 4)
ml-copilot deploy --target sagemaker
```

## How it works

Built on [agentlite](https://github.com/hakansabunis/agentlite) — a
small Claude agent library — ml-copilot uses one orchestrator agent
per command, plus focused sub-agents for sub-tasks:

```
       cli.py
         │
   ┌─────┴─────┐
   ▼           ▼
 advise      watch                ... deploy
 agent       agent
   │           │
   ▼           ▼
 ModelAdvisor  MetricsWatcher (Haiku, polls)
  (Opus)       Diagnostician  (Opus, called on anomaly)
```

Every action that would modify your code, config, or run a training
process **asks permission first** — agentlite's permission system is
first-class, not an afterthought.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Project context

Each ml-copilot project keeps a small folder, similar in spirit to
`.git/`:

```
.mlcopilot/
├── project.yaml        # metadata
├── context.json        # decisions, recommendations, active state
├── datasets/           # registered datasets
└── runs/               # training run history
```

This is what makes ml-copilot more than a chat tool: by the time you
run `deploy`, every earlier decision is still in memory.

## Roadmap

| Phase           | Commands                              |  Status       |
| --------------- | ------------------------------------- | :-----------: |
| **Faz 1 (v0.1)**| `init`, `advise`                      | 🚧 In progress |
| **Faz 2 (v0.2)**| `audit`, `watch`, `compare`           | 📅 Planned    |
| **Faz 3 (v0.3)**| `evaluate`                            | 📅 Planned    |
| **Faz 4 (v0.4)**| `deploy`                              | 📅 Planned    |

See [CHANGELOG.md](CHANGELOG.md) for detailed plans and
[ARCHITECTURE.md](ARCHITECTURE.md) for the design.

## Non-goals

To stay focused, ml-copilot will **not** try to be:

- **AutoML** (use AutoGluon, AutoSklearn)
- **Experiment tracker** (use MLflow, W&B)
- **Code assistant** (use Cursor, Copilot, aider)
- **Monitoring dashboard** (use Grafana, Streamlit)

ml-copilot **advises**; you decide.

## Contributing

Pre-alpha — issues and discussions welcome, PRs after v0.1.

## License

MIT © 2026 Hakan Sabunis
