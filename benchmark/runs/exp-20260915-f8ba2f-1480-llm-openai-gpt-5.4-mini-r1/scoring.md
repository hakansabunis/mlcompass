# Scoring — exp-20260915-f8ba2f-1480-llm-openai-gpt-5.4-mini-r1

Dataset: openml-1480 (ilpd)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1480']
missed_issues: 0  []
unverified_findings: 9

## Findings reported by the tool

- 13 exact duplicate row(s) (2.2% of the data). A random split puts copies of
- V2 → one-hot encode Male/Female; optionally add a binary sex indicator V2 is the only categorical predictor and is low-cardinality, so encoding
- V3,V4,V5,V6,V7 → apply robust scaling or log1p/quantile transforms; These columns show many outliers and heavy right tails (especially V3, V4,
- V3/V4/V5/V6/V7 → create ratio and difference features such as V3/V4, V5/V6, The dataset contains multiple related numeric measures with likely latent
- V1 → treat as a bounded numeric feature and consider age bands or V1 has a plausible age-like distribution (4 to 90) with a broad spread, so
- all numeric columns → standardize for linear models and add missingness Current data have no missing values, but scaling will help logistic
- Duplicate rows (13 exact duplicates, 2.2%) → Verify whether duplicates are legitimate repeated measurements; if not,
- Class imbalance (Class 1: 71.4%, Class 2: 28.6%) → Use stratified splits, emphasize balanced metrics such as
- Small dataset size (583 rows) → Prefer simpler models and nested or repeated cross-validation; keep
- Outlier-heavy numeric features (especially V3, V4, V5, V6, V7) → Inspect extreme values for data-entry issues, and test robust
- Potential target leakage from preprocessing done before splitting → Fit encoders, scalers, imputers, and any outlier handling only on

## Unmatched findings (unverified, not refuted)

- V2 → one-hot encode Male/Female; optionally add a binary sex indicator V2 is the only categorical predictor and is low-cardinality, so encoding
- V3,V4,V5,V6,V7 → apply robust scaling or log1p/quantile transforms; These columns show many outliers and heavy right tails (especially V3, V4,
- V3/V4/V5/V6/V7 → create ratio and difference features such as V3/V4, V5/V6, The dataset contains multiple related numeric measures with likely latent
- V1 → treat as a bounded numeric feature and consider age bands or V1 has a plausible age-like distribution (4 to 90) with a broad spread, so
- all numeric columns → standardize for linear models and add missingness Current data have no missing values, but scaling will help logistic
- Class imbalance (Class 1: 71.4%, Class 2: 28.6%) → Use stratified splits, emphasize balanced metrics such as
- Small dataset size (583 rows) → Prefer simpler models and nested or repeated cross-validation; keep
- Outlier-heavy numeric features (especially V3, V4, V5, V6, V7) → Inspect extreme values for data-entry issues, and test robust
- Potential target leakage from preprocessing done before splitting → Fit encoders, scalers, imputers, and any outlier handling only on

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
