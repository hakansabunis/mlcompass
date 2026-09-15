# Scoring — exp-20260915-eb4c19-1464-llm-ollama-qwen2.5-7b-r1

Dataset: openml-1464 (blood-transfusion-service-center)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1464']
missed_issues: 0  []
unverified_findings: 5

## Findings reported by the tool

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies
- V1 → log transformation to manage heavy tails and outliers Heavy standard deviation suggests skewed distribution.
- V2 → log transformation to manage heavy tails and outliers High standard deviation and number of outliers indicate a skewed
- V3 → log1p or sqrt for skewed data, scale for larger effect Extremely wide range with outliers can benefit from transformations
- V4 → standard scaling if on same physical units; otherwise, log for Value looks reasonable without extreme outliers
- Class imbalance (76.2% positive class) → Ensure model performance is evaluated with balanced accuracy or F1
- Duplicate rows (28.7%) → Remove duplicates before training the dataset.

## Unmatched findings (unverified, not refuted)

- V1 → log transformation to manage heavy tails and outliers Heavy standard deviation suggests skewed distribution.
- V2 → log transformation to manage heavy tails and outliers High standard deviation and number of outliers indicate a skewed
- V3 → log1p or sqrt for skewed data, scale for larger effect Extremely wide range with outliers can benefit from transformations
- V4 → standard scaling if on same physical units; otherwise, log for Value looks reasonable without extreme outliers
- Class imbalance (76.2% positive class) → Ensure model performance is evaluated with balanced accuracy or F1

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
