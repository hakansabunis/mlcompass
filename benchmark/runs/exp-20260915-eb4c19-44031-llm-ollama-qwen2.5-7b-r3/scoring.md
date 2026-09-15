# Scoring — exp-20260915-eb4c19-44031-llm-ollama-qwen2.5-7b-r3

Dataset: openml-44031 (california)
Configuration: llm:ollama-qwen2.5-7b
Status: completed
Scoring version: 2.0; ground-truth registry 1.1

Scope scored: LLM output only, from the first of ✨ Recommended models, 🔧 Feature engineering, ⚠ Pitfalls onward.

known_issues: 1
correct_detections: 0  []
missed_issues: 1  ['CENSOR-44031']
unverified_findings: 5  (= 2 unmatched detection claims + 3 suggestions)

A detection claim asserts a defect and is checked against the registry;
it comes from `⚠ Warnings` or `⚠ Pitfalls`. A suggestion comes from
`🔧 Feature engineering` or `✨ Recommended models` and has no ground
truth to match against, so it is counted apart rather than pooled with
claims that could have matched and did not.

## Findings in scope

- AveRooms → Remove outliers before deriving features to avoid skewing the AveRooms has many outliers that could distort feature engineering if not
- AveBedrms → Normalization or removing outliers will help maintain a stable The presence of numerous outliers in AveBedrms suggests normalizing the
- Population → Consider log transformation to reduce skewness and stabilize Population has a high standard deviation indicating potential skewness
- Target variable is capped at its maximum value → Analyze the data source to understand why the target is capped, and
- Outliers in 'AveRooms' and 'AveBedrms' → Apply appropriate outlier detection and mitigation strategies before

## Unmatched detection claims (unverified, not refuted)

- Target variable is capped at its maximum value → Analyze the data source to understand why the target is capped, and
- Outliers in 'AveRooms' and 'AveBedrms' → Apply appropriate outlier detection and mitigation strategies before

## Suggestions (no ground truth to match against)

- AveRooms → Remove outliers before deriving features to avoid skewing the AveRooms has many outliers that could distort feature engineering if not
- AveBedrms → Normalization or removing outliers will help maintain a stable The presence of numerous outliers in AveBedrms suggests normalizing the
- Population → Consider log transformation to reduce skewness and stabilize Population has a high standard deviation indicating potential skewness

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['CENSOR-44031']

- Target 'price' is piled up at its maximum: 965 row(s) (4.7%) sit at exactly

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
