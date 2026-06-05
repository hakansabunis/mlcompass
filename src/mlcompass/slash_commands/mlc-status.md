---
description: Show the active mlcompass project status (decisions, command counts, active state)
---

Use the `mlcompass_status` tool to summarise the current project context. If no `$ARGUMENTS` are given, pass `project_path="."` so it walks up from the current directory.

After the tool returns, in the user's language:

1. Project metadata: name, when it was created, mlcompass version that initialised it.
2. Active state: `project_type`, `target_column`, `active_dataset`, `current_run`. Note if any are null — that's a signal the user hasn't run `mlc-advise` yet in this project.
3. Command activity counts (e.g. `advise: 2, audit: 1, evaluate: 3`).
4. The last N decisions in a short table.

If the result includes a `hints` field, surface it — that's where mlcompass tells the user about non-obvious state (e.g. "you ran tools before init, those calls weren't logged").
