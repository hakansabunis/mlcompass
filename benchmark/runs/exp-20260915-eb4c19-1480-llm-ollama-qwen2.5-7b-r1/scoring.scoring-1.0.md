# Scoring — exp-20260915-eb4c19-1480-llm-ollama-qwen2.5-7b-r1

Dataset: openml-1480 (ilpd)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1480']
missed_issues: 0  []
unverified_findings: 4

## Findings reported by the tool

- 13 exact duplicate row(s) (2.2% of the data). A random split puts copies of
- V3 → log(V3) or Standardize High variance, logarithmic transformation might help
- V4 → log(V4) or Standardize High variance, logarithmic transformation might help
- V5 → log(V5) or Winsorization High variance and outliers, winzorizing the high values could be useful
- Class imbalance (71.4% of one class) → Use AUC/F1 as metrics; apply SMOTE to generate synthetic data or use
- Duplicate rows → Deduplicate the dataset before splitting into training and test sets

## Unmatched findings (unverified, not refuted)

- V3 → log(V3) or Standardize High variance, logarithmic transformation might help
- V4 → log(V4) or Standardize High variance, logarithmic transformation might help
- V5 → log(V5) or Winsorization High variance and outliers, winzorizing the high values could be useful
- Class imbalance (71.4% of one class) → Use AUC/F1 as metrics; apply SMOTE to generate synthetic data or use

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
