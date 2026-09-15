# Scoring — exp-20260915-eb4c19-1480-llm-ollama-qwen2.5-7b-r2

Dataset: openml-1480 (ilpd)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1480']
missed_issues: 0  []
unverified_findings: 5

## Findings reported by the tool

- 13 exact duplicate row(s) (2.2% of the data). A random split puts copies of
- V3 → log, clip or normalize to reduce outliers Logarithmic transformation can help manage extreme values
- V4 → log, clip or normalize to reduce outliers Same reason as V3
- V5 → log, clip or normalize to reduce outliers Same reason as V3 and V4
- V6 → log, clip or normalize to reduce outliers Same reason as V3
- Class imbalance (71.36% vs 28.64%) → Use AUC/F1, class_weight='balanced', or focal loss
- Exact duplicate rows detected → Remove duplicates before training

## Unmatched findings (unverified, not refuted)

- V3 → log, clip or normalize to reduce outliers Logarithmic transformation can help manage extreme values
- V4 → log, clip or normalize to reduce outliers Same reason as V3
- V5 → log, clip or normalize to reduce outliers Same reason as V3 and V4
- V6 → log, clip or normalize to reduce outliers Same reason as V3
- Class imbalance (71.36% vs 28.64%) → Use AUC/F1, class_weight='balanced', or focal loss

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
