# Scoring — exp-20260915-df0632-1480-llm-ollama-qwen2.5-7b-r1

Dataset: openml-1480 (ilpd)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1480']
missed_issues: 0  []
unverified_findings: 4

## Findings reported by the tool

- 13 exact duplicate row(s) (2.2% of the data). A random split puts copies of
- V3 → Log the values (np.log1p) to reduce skewing effect High standard deviation and IQR count suggest a potential power-law
- V4 → Normalize with StandardScaler or MinMaxScaler as per sklearn docs Feature has extreme outliers, which can be detrimental to some algorithms
- V5 → Apply Box-Cox transformation if normality is not met; otherwise, log High standard deviation and IQR count suggest heavy skewing
- Class imbalance (71.36% vs 28.64%) in target variable → Use AUC/F1, class_weight='balanced', or custom sampling techniques

## Unmatched findings (unverified, not refuted)

- V3 → Log the values (np.log1p) to reduce skewing effect High standard deviation and IQR count suggest a potential power-law
- V4 → Normalize with StandardScaler or MinMaxScaler as per sklearn docs Feature has extreme outliers, which can be detrimental to some algorithms
- V5 → Apply Box-Cox transformation if normality is not met; otherwise, log High standard deviation and IQR count suggest heavy skewing
- Class imbalance (71.36% vs 28.64%) in target variable → Use AUC/F1, class_weight='balanced', or custom sampling techniques

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
