---
description: Analyze a dataset and recommend models, features, and data-quality fixes
---

Use the `mlcompass_advise` tool to analyze the dataset at: $ARGUMENTS

After the tool returns, summarize the result in the user's language:

1. Dataset shape (rows × columns) and detected file format.
2. Detected target column and confidence level. If confidence is low or null, suggest passing `target=<name>` explicitly.
3. Inferred task type (binary classification / multiclass / regression).
4. The three most important warnings (e.g. high-missing columns, ID columns, class imbalance, sparse numerics, dirty-string columns).
5. One sentence on what to do next (e.g. "drop these columns", "impute these", "consider a log transform").

If the dataset has obvious data-quality issues, end with: "Want me to write a small cleaning script before we move on?"
