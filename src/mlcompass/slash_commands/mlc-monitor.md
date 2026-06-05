---
description: Compare a reference dataset to a current one and detect drift (PSI / KS / chi²)
---

This command needs two file paths: a reference dataset (training-era / baseline) and a current dataset (production / new). Parse them from: $ARGUMENTS

`$ARGUMENTS` should look like: `reference.csv current.csv` (space-separated).

Since `mlcompass_monitor` is exposed via the CLI but not the MCP server in v0.7.x, **fall back to the CLI**:

```
mlcompass monitor <reference> <current>
```

Run the command via `Bash`. After it returns, in the user's language:

1. Per-feature drift summary: aggregate PSI, max PSI, count of stable / moderate / major features.
2. Top drifted features by PSI.
3. Verdict: stable / moderate_drift / major_drift.
4. Whether a retrain is recommended.

If exit code is 1, that's intentional — major drift triggered a CI-gateable failure. Not a bug.
