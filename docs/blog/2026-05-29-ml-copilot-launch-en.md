# Launching ml-copilot: an LLM agent for the full ML pipeline

*2026-05-29 — Hakan Sabunis*

I just shipped the first public version of **ml-copilot**, a small CLI
agent that sits next to you through your whole ML pipeline. It's MIT,
written in Python, and runs on top of Claude.

```bash
pip install ml-copilot
ml-copilot init churn-project
ml-copilot advise data.csv --target churn
```

This post is the story of why I built it, what's in v0.1, and where it's
going.

## The motivation

I'm a CS student finishing my Capstone on flood prediction for Turkey.
The model is an LSTM trained on years of weather, river discharge, and
geographic data. It works — but getting there took dozens of training
runs, and the same kinds of mistakes kept coming back:

- Wrong loss for an imbalanced problem (using BCE instead of focal loss
  while the positive class is 12% of the data)
- Forgetting `torch.manual_seed(...)` so results don't reproduce
- A validation split that's too small to give a stable signal
- Numeric instability (`log(0)` after a sigmoid) showing up as NaN at
  epoch 14 of a 6-hour run

Every one of these is the kind of thing a senior engineer next to you
would catch in 30 seconds. Existing ML tools — W&B, TensorBoard, MLflow,
Comet — are great at logging what happened. None of them advise.
Frameworks like AutoSklearn and AutoGluon take the other extreme:
they're fully autonomous, so you lose the ability to learn from the
choices being made.

I wanted something in between. A tool that watches what you're doing,
explains what it sees, and asks before changing anything.

## What v0.1 ships

The first release is the pre-training half of the pipeline: **advise**
mode. You point it at a CSV and it produces a structured recommendation.

```
┌───────────── 📊 Dataset analysis ─────────────┐
│ Path:    examples/customer_churn.csv          │
│ Shape:   500 rows × 8 columns                 │
│ Target:  churn (high confidence)              │
│ Task:    binary classification (0=98%, 1=2%)  │
└───────────────────────────────────────────────┘

⚠ Warnings
  • Class imbalance detected (1.6% minority class). Don't optimise
    accuracy — use AUC/F1/recall@k. Consider class_weight='balanced'
    or focal loss.

✨ Recommended models
  • XGBoost              AUC 0.80 – 0.84  — strong tabular baseline
  • Logistic Regression  AUC 0.72 – 0.76  — interpretable baseline
  • LightGBM             AUC 0.80 – 0.85  — faster than XGB at this size

🔧 Feature engineering
  • signup_date  → derive days_since_signup, dayofweek
  • country (30 cats) → target encoding or top-N grouping
  • support_calls  → log1p transform (zero-inflated)

⚠ Pitfalls
  • Severe class imbalance — accuracy will mislead you
  • 30 unique countries — risk of overfitting with naive encoding
```

The interesting design choice: the dataset analyzer is **pure pandas**.
There's no LLM round-trip to ask Claude "what type is this column".
That keeps `advise` fast and predictable. The LLM only kicks in to do
the actual recommendation, given the structured analysis as input.

## What the v0.1 architecture looks like

```
            CLI dispatcher (cli.py)
                    │
              ┌─────┴──────┐
              ▼            ▼
            init        advise
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
        analyze_dataset  ModelAdvisor  ProjectContext
        (pure pandas)    (Claude Opus) (.mlcopilot/)
                            │
                            ▼
                       Rich UI
                       (terminal)
```

A persistent `.mlcopilot/` directory carries context across commands:
which dataset is active, which task type, which decisions you've made
and why. This is what will let the planned later commands —
`audit`, `watch`, `evaluate`, `deploy` — know what you already chose.

ml-copilot is built on **[agentlite](https://github.com/hakansabunis/agentlite)**,
a ~2K-line Claude agent library I wrote for the same Capstone project.
That gives us:

- **Prompt caching by default** — the system prompt for the advisor
  is large, and we don't pay to re-send it every call
- **First-class permissions** — every tool that touches your code or
  config is `requires_confirmation=True` by design
- **Sub-agent factory** — when the planned `watch` mode catches a
  problem, the diagnostician is a sub-agent spawned only on demand,
  not always-on

## What's coming next

v0.2 ships the part that originally motivated the project: live
training watching.

- `ml-copilot audit <script>` — static analysis of your training script
- `ml-copilot watch <script>` — live monitor that detects plateau,
  overfit, and NaN and asks permission before suggesting a config edit
- `ml-copilot compare run-a run-b` — LLM-explained run diff

After that, v0.3 adds `evaluate` (post-training analysis), and v0.4
adds `deploy` (deployment readiness check).

The full roadmap is in [CHANGELOG.md](https://github.com/hakansabunis/ml-copilot/blob/main/CHANGELOG.md).
The design rationale is in [ARCHITECTURE.md](https://github.com/hakansabunis/ml-copilot/blob/main/ARCHITECTURE.md).

## Try it

```bash
pip install ml-copilot
ml-copilot init demo
ml-copilot advise <your.csv>
```

There's no real LLM call unless `ANTHROPIC_API_KEY` is set, so you can
explore the deterministic analyzer for free by using `--no-llm`.

Three example datasets live in `examples/` if you don't have one handy.

## Feedback welcome

ml-copilot is alpha. The advisor prompt is the thing I expect to iterate
on the most — if you try it and the recommendation is off, please open
an issue with the dataset (or a minimal anonymised version) and what
you'd have hoped to see.

GitHub: <https://github.com/hakansabunis/ml-copilot>
