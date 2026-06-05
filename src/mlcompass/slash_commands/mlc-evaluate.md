---
description: Score a predictions table (metrics, threshold sweep, confusion matrix, leakage smell)
---

Use the `mlcompass_evaluate` tool on the predictions file at: $ARGUMENTS

Do NOT pass `task=` — the tool auto-detects binary / multiclass / regression from the columns.

After the tool returns, summarize in the user's language:

1. Detected task type and headline metrics (accuracy, F1, AUC, R², MAE — whichever are relevant).
2. Confusion matrix (binary: 2×2, multiclass: K×K).
3. For binary: best F1 threshold from the threshold sweep, and what changes at that threshold.
4. For regression: residual mean/std and the top-K worst-error rows.

If the result includes a `leakage_investigation` field (the metric tripped the smell threshold), call it out IMMEDIATELY with a 🔬 prefix. Read the panel carefully and list the candidate leak columns by name + the correlation method that flagged them. Do NOT speculate beyond what the panel contains — that's the whole point of the anti-hallucination contract.
