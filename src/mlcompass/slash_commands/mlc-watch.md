---
description: Scan a training log (plain text / TensorBoard / W&B) for plateau, divergence, overfit, NaN
---

Use the `mlcompass_watch` tool on the training log at: $ARGUMENTS

The tool auto-detects the source type (plain text, TensorBoard event file, or W&B local run dir). After the tool returns, summarize in the user's language:

1. Source type detected and number of snapshots parsed.
2. Last epoch / step observed.
3. Findings list, grouped by severity. Common detectors: `nan`, `divergence`, `plateau`, `overfitting`.
4. For each finding, name the metric involved and the epoch where it became suspicious.
5. End with a "what this usually means and what to try" paragraph.

If no findings, say "training looks healthy through epoch N" and stop.
