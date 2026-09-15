# Scoring — exp-20260915-f8ba2f-1464-llm-mistral-ministral-8b-r1

Dataset: openml-1464 (blood-transfusion-service-center)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1464']
missed_issues: 0  []
unverified_findings: 7

## Findings reported by the tool

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies
- V1 → Apply log-transform (e.g., `log(V1 + 1)`) to reduce skew and outlier High zero ratio (0.67%) and outliers (IQR/Z-score) suggest right-skewed
- V2 → Bin into quantile-based buckets (e.g., 4 groups) to reduce noise from Extreme outliers (45 IQR, 12 Z-score) may dominate model weights; binning
- V3 → Normalize (e.g., `MinMaxScaler`) or log-transform; derive `V3/V4` Massive scale (250–12500) and outliers (45 IQR) suggest normalization;
- V4 → No transformation needed (no outliers), but consider interaction terms Normal distribution (no IQR/Z-score outliers) and moderate spread; may
- Class → Resample (SMOTE/ADASYN) or use class weights Severe imbalance (76%/24%) risks bias toward majority class.
- Duplicate rows (215, 28.7% of data) → De-duplicate via `drop_duplicates()` before train-test split, or split
- Outliers in V1, V2, V3 → Validate if outliers are genuine (e.g., measurement errors) or
- No feature names or domain context → Label columns (e.g., `V1` → `donations_last_month`) to guide feature

## Unmatched findings (unverified, not refuted)

- V1 → Apply log-transform (e.g., `log(V1 + 1)`) to reduce skew and outlier High zero ratio (0.67%) and outliers (IQR/Z-score) suggest right-skewed
- V2 → Bin into quantile-based buckets (e.g., 4 groups) to reduce noise from Extreme outliers (45 IQR, 12 Z-score) may dominate model weights; binning
- V3 → Normalize (e.g., `MinMaxScaler`) or log-transform; derive `V3/V4` Massive scale (250–12500) and outliers (45 IQR) suggest normalization;
- V4 → No transformation needed (no outliers), but consider interaction terms Normal distribution (no IQR/Z-score outliers) and moderate spread; may
- Class → Resample (SMOTE/ADASYN) or use class weights Severe imbalance (76%/24%) risks bias toward majority class.
- Outliers in V1, V2, V3 → Validate if outliers are genuine (e.g., measurement errors) or
- No feature names or domain context → Label columns (e.g., `V1` → `donations_last_month`) to guide feature

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
