---
description: Statically audit a Python training script for common ML mistakes
---

Use the `mlcompass_audit` tool to statically analyse the Python script at: $ARGUMENTS

After the tool returns, present findings in the user's language:

1. Group findings by severity: errors first, then warnings, then info.
2. For each finding, give: rule id, line number, the one-sentence explanation, and the realistic worst-case if it's ignored.
3. End with a prioritized "fix this first" list — the two or three things most likely to silently ruin a training run.

If the script has no errors, say so clearly and move on.

If there are at least two errors, ask: "Want me to write the patch?"
