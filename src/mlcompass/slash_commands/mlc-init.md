---
description: Initialize a new mlcompass project in the current directory
---

Use the `mlcompass_init` tool to create a new project. Parse the project name from `$ARGUMENTS`. If no name is given, ask the user for one before calling the tool.

After the tool returns:

1. Confirm the project was created and report the `.mlcompass/` directory path.
2. Suggest a logical next step: "Now run `/mlc-advise <your-data.csv>` to register your dataset."

If the tool returns `ProjectExistsError`, do not retry. Tell the user a project already exists, and use `/mlc-status` to inspect it.
