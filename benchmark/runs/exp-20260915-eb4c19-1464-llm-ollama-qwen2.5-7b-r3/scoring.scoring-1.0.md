# Scoring — exp-20260915-eb4c19-1464-llm-ollama-qwen2.5-7b-r3

Dataset: openml-1464 (blood-transfusion-service-center)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1464']
missed_issues: 0  []
unverified_findings: 5

## Findings reported by the tool

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies
- V1 → log(V1 + 1) Log transformation can stabilize variance for skewed data.
- V2 → z_score(V2) Z-score normalization helps in comparing variables with different scales.
- V3 → log(V3 + 1) V3 has a wide range, log transformation can linearize the relationship and
- V4 → z_score(V4) Similar to V2, z-score normalization helps in handling scale differences
- Class imbalance (76.2% vs 23.8%) → Use AUC/F1, class_weight='balanced', or focal loss
- Possible duplicate rows (28.7%) → Deduplicate data before splitting into train and test sets

## Unmatched findings (unverified, not refuted)

- V1 → log(V1 + 1) Log transformation can stabilize variance for skewed data.
- V2 → z_score(V2) Z-score normalization helps in comparing variables with different scales.
- V3 → log(V3 + 1) V3 has a wide range, log transformation can linearize the relationship and
- V4 → z_score(V4) Similar to V2, z-score normalization helps in handling scale differences
- Class imbalance (76.2% vs 23.8%) → Use AUC/F1, class_weight='balanced', or focal loss

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
