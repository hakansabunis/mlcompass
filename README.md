# ml-copilot

> An LLM agent that watches your ML training and tells you what's wrong.

[![PyPI](https://img.shields.io/pypi/v/ml-copilot.svg)](https://pypi.org/project/ml-copilot/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

🚧 **Pre-alpha (v0.0.1)** — under active development. APIs will change before v0.1.

## What it does

You run `train.py`. ml-copilot watches in your terminal:

- Reads your TensorBoard / W&B / plain-text logs in real time
- Detects plateaus, overfitting, NaN, divergence
- Reads your training script and diagnoses the likely cause
- Asks permission before changing anything

Think of it as a senior ML engineer sitting next to you.

## Quick example

```bash
ml-copilot watch train.py
```

Output after 8 epochs:

```
⚠️  Epoch 8 — overfitting detected
   Train loss: 0.118  |  Val loss: 0.387  (gap 0.27, normal <0.1)

   Likely cause: regularization is too weak for the model capacity.

   Suggested fix: increase dropout 0.1 → 0.3
   Apply and restart training? [y/N]
```

## Why ml-copilot

Existing tools **log**; ml-copilot **advises**:

|                                | W&B / TensorBoard | Cursor / Devin | **ml-copilot** |
| ------------------------------ | :---------------: | :------------: | :------------: |
| Logs metrics                   |        ✅         |       ❌       |       ✅       |
| Watches training in real time  |        ❌         |       ❌       |       ✅       |
| Diagnoses problems proactively |        ❌         |    reactive    |       ✅       |
| Suggests training-aware fixes  |        ❌         |   code-level   |       ✅       |
| Permission-gated actions       |        ❌         |    partial     |   first-class  |

## Install

```bash
pip install ml-copilot
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

### Watch a live training run

```bash
ml-copilot watch train.py
```

### Audit a script (one-shot static analysis)

```bash
ml-copilot audit train.py
```

### Compare two runs

```bash
ml-copilot compare runs/run-3 runs/run-7
```

## How it works

Built on [agentlite](https://github.com/hakansabunis/agentlite). Architecture:

```
You (terminal)
    │
    ▼
ml-copilot (orchestrator agent, Opus)
    ├── Watcher sub-agent       (Haiku, runs continuously)
    ├── Diagnostician sub-agent (Opus, called on anomaly)
    └── Code Inspector sub-agent (Haiku, reads train.py)
    │
    ▼ permission-gated actions
[edit_config | restart_training | apply_code_patch]
```

Every action that would change your code, config, or process **asks permission first.**

## Status

| Feature                                   |   Status        |
| ----------------------------------------- | :-------------: |
| `audit` mode (static script analysis)     | 🚧 In progress  |
| `watch` mode (live monitoring)            | 🚧 In progress  |
| Plateau / overfitting / NaN detection     | 🚧 In progress  |
| `compare` mode (run diff)                 | 📅 Planned      |
| Auto-restart with config edit             | 📅 Planned      |
| Jupyter magic (`%mlc watch`)              | 📅 Planned      |
| W&B API integration                       | 📅 Planned      |

## Contributing

Pre-alpha — issues and discussions welcome, PRs after v0.1.

## License

MIT © 2026 Hakan Sabunis
