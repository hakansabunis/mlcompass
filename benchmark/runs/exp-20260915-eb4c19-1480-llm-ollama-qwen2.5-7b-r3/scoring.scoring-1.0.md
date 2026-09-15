# Scoring — exp-20260915-eb4c19-1480-llm-ollama-qwen2.5-7b-r3

Dataset: openml-1480 (ilpd)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1480']
missed_issues: 0  []
unverified_findings: 5

## Findings reported by the tool

- 13 exact duplicate row(s) (2.2% of the data). A random split puts copies of
- V3 → log1p(V3) to linearize positive right-skewed distribution, handling Outliers exist (84 iqr_count).
- V4 → log1p(V4) for similar reasons as V3 Outliers exist (81 iqr_count)
- V5 → scale(V5) to maintain feature range, which can be important for High std and outliers present(69 iqr_count).
- V7 → log1p(V7) for similar reasons as V3 and V4 Outliers exist (66 iqr_count).
- Class Imbalance (71% vs. 29%) → Use AUC or F1 score, class_weight='balanced', or consider oversampling
- Duplicate rows (2.2% of the data) → Remove exact duplicates before splitting into train and test sets to

## Unmatched findings (unverified, not refuted)

- V3 → log1p(V3) to linearize positive right-skewed distribution, handling Outliers exist (84 iqr_count).
- V4 → log1p(V4) for similar reasons as V3 Outliers exist (81 iqr_count)
- V5 → scale(V5) to maintain feature range, which can be important for High std and outliers present(69 iqr_count).
- V7 → log1p(V7) for similar reasons as V3 and V4 Outliers exist (66 iqr_count).
- Class Imbalance (71% vs. 29%) → Use AUC or F1 score, class_weight='balanced', or consider oversampling

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
