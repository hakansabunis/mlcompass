---
description: Side-by-side comparison of two training runs (config diff + final-metric winner)
---

Use the `mlcompass_compare` tool with the two run identifiers parsed from: $ARGUMENTS

`$ARGUMENTS` should contain two run names or paths separated by a space (e.g. `baseline lower-lr`). If a project is active in the current directory, pass `project_path="."` so name-based lookups work.

After the tool returns, in the user's language:

1. Show the config diff in a small table.
2. Show the final-metric comparison: which run won each metric, by how much.
3. State the overall verdict (`A wins`, `B wins`, `mixed`, or `tie`).
4. Important: if the verdict is `mixed` and run A only wins on training metrics, warn the user that they may be reading the wrong signal — model selection should follow validation, not training.
