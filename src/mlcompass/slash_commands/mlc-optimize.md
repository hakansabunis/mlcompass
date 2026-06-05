---
description: Analyse run history and recommend the next hyperparameter configurations to try
---

This command analyses a run history under `<project>/runs/` and proposes the next configurations.

Parse `$ARGUMENTS` for a metric name (required) and optional `direction` (`max` or `min`, default `max`). Example: `val_acc max` or `val_loss min`.

Since `mlcompass_optimize` is exposed via the CLI but not the MCP server in v0.7.x, **fall back to the CLI**:

```
mlcompass optimize --metric <name> --direction <max|min>
```

Run the command via `Bash`. After it returns, in the user's language:

1. Best run so far (id, score, key hyperparameters).
2. Sensitivity table: which hyperparameters have the strongest correlation with the metric.
3. The N suggested next configurations, each with a one-sentence rationale.

Highlight if a suggested value crosses a domain-aware soft cap (e.g. dropout = 0.8 means "we capped this because >0.8 usually destroys signal").

If there are fewer than 3 runs in the history, the sensitivity analysis is unreliable; tell the user "we need more runs first" and suggest two or three reasonable starting configurations from scratch.
