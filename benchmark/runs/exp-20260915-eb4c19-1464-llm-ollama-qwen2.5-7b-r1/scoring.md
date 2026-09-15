# Scoring — exp-20260915-eb4c19-1464-llm-ollama-qwen2.5-7b-r1

Dataset: openml-1464 (blood-transfusion-service-center)
Configuration: llm:ollama-qwen2.5-7b
Status: completed
Scoring version: 2.0; ground-truth registry 1.1

Scope scored: LLM output only, from the first of ✨ Recommended models, 🔧 Feature engineering, ⚠ Pitfalls onward.

known_issues: 1
correct_detections: 0  []
missed_issues: 1  ['DUP-1464']
unverified_findings: 6  (= 2 unmatched detection claims + 4 suggestions)

A detection claim asserts a defect and is checked against the registry;
it comes from `⚠ Warnings` or `⚠ Pitfalls`. A suggestion comes from
`🔧 Feature engineering` or `✨ Recommended models` and has no ground
truth to match against, so it is counted apart rather than pooled with
claims that could have matched and did not.

## Findings in scope

- V1 → log transformation to manage heavy tails and outliers Heavy standard deviation suggests skewed distribution.
- V2 → log transformation to manage heavy tails and outliers High standard deviation and number of outliers indicate a skewed
- V3 → log1p or sqrt for skewed data, scale for larger effect Extremely wide range with outliers can benefit from transformations
- V4 → standard scaling if on same physical units; otherwise, log for Value looks reasonable without extreme outliers
- Class imbalance (76.2% positive class) → Ensure model performance is evaluated with balanced accuracy or F1
- Duplicate rows (28.7%) → Remove duplicates before training the dataset.

## Unmatched detection claims (unverified, not refuted)

- Class imbalance (76.2% positive class) → Ensure model performance is evaluated with balanced accuracy or F1
- Duplicate rows (28.7%) → Remove duplicates before training the dataset.

## Suggestions (no ground truth to match against)

- V1 → log transformation to manage heavy tails and outliers Heavy standard deviation suggests skewed distribution.
- V2 → log transformation to manage heavy tails and outliers High standard deviation and number of outliers indicate a skewed
- V3 → log1p or sqrt for skewed data, scale for larger effect Extremely wide range with outliers can benefit from transformations
- V4 → standard scaling if on same physical units; otherwise, log for Value looks reasonable without extreme outliers

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['DUP-1464']

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
