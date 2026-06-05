---
description: Focused leakage check — run evaluate, then narrate ONLY the leakage panel
---

Use the `mlcompass_evaluate` tool on the predictions file at: $ARGUMENTS

You're running this command because the user suspects data leakage. Their attention is on the `leakage_investigation` field, not the regular metrics.

After the tool returns:

1. If `leakage_investigation` is **absent** (no smell threshold tripped), say so plainly: "metrics are below the leakage-smell threshold; no investigation was triggered. The headline values were ..."  and report the metrics in one line. Then stop.

2. If `leakage_investigation` is **present**:
   - Lead with the suspicious metric value.
   - List `candidate_leak_columns` by name. For each, name the correlation method that flagged it (Pearson or Spearman) and the value.
   - Report `perfect_match_rate`. If ≥ 0.95, that's a smoking gun.
   - State the verdict: leakage_likely / leakage_uncertain / score_legitimate / cannot_determine.

Strict rule: cite ONLY what's in the panel. If a column doesn't appear in `candidate_leak_columns`, do not speculate about it.

End with a short manual checklist (max 3 items) the user should perform — never code patches.
