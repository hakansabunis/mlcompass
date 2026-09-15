# Scoring — exp-20260915-df0632-1464-llm-ollama-qwen2.5-7b-r1

Dataset: openml-1464 (blood-transfusion-service-center)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1464']
missed_issues: 0  []
unverified_findings: 3

## Findings reported by the tool

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies
- V1 → log-transform, bin into quartiles/buckets ... reduces skew and handles outliers more gracefully
- V2 → standardize before applying log or sqrt transformation ... helps with outliers without skewing the data too much
- High class imbalance (76.2% vs 23.8%) → Use AUC, F1 score, or apply class weights
- Exact duplicate rows present in dataset → Remove duplicates before splitting into train/test

## Unmatched findings (unverified, not refuted)

- V1 → log-transform, bin into quartiles/buckets ... reduces skew and handles outliers more gracefully
- V2 → standardize before applying log or sqrt transformation ... helps with outliers without skewing the data too much
- High class imbalance (76.2% vs 23.8%) → Use AUC, F1 score, or apply class weights

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
