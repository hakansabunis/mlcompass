# Scoring — exp-20260915-eb4c19-1464-llm-ollama-qwen2.5-7b-r2

Dataset: openml-1464 (blood-transfusion-service-center)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1464']
missed_issues: 0  []
unverified_findings: 5

## Findings reported by the tool

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies
- V1 → normalize or standardize ... wide range, might benefit from scaling for consistent learning
- V2 → log transformation ... skewed distribution with outliers, could help stabilize variance and
- V3 → scale or normalize ... very wide range compared to others, benefits from scaling for
- V4 → quantization or binning ... few outliers already removed, potential value in discretizing into
- Class imbalance (76.2% vs 23.8%) → use AUC/F1, class_weight='balanced', or focal loss to handle imbalance
- Highly exact duplicate rows (28.7%) → de-duplicate the data before splitting into training and testing sets

## Unmatched findings (unverified, not refuted)

- V1 → normalize or standardize ... wide range, might benefit from scaling for consistent learning
- V2 → log transformation ... skewed distribution with outliers, could help stabilize variance and
- V3 → scale or normalize ... very wide range compared to others, benefits from scaling for
- V4 → quantization or binning ... few outliers already removed, potential value in discretizing into
- Class imbalance (76.2% vs 23.8%) → use AUC/F1, class_weight='balanced', or focal loss to handle imbalance

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
