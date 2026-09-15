# Scoring — exp-20260915-eb4c19-1464-llm-mistral-ministral-8b-r2

Dataset: openml-1464 (blood-transfusion-service-center)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1464']
missed_issues: 0  []
unverified_findings: 8

## Findings reported by the tool

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies
- V1 → Apply log transformation (e.g., `log(V1 + 1)`) or bin into quantiles High zero ratio (0.67%) and right-skewed distribution (mean > median)
- V2 → Create interaction terms with `V1` (e.g., `V1 * V2`) or normalize by Both `V2` and `V3` have extreme outliers (IQR-based) and may interact
- V3 → Scale to a meaningful range (e.g., `V3 / 1000`) or flag extreme values Extreme skewness and outliers may dominate model weights; scaling or
- V4 → No transformation needed, but consider binning into clinically Normal distribution (no outliers) and moderate spread; binning could
- Class → Resample (oversample class `2` or undersample class `1`) or use Severe imbalance (76% vs. 24%) risks biased models; mitigation is critical
- Duplicate rows (215 exact duplicates, 28.7% of data) → De-duplicate using a unique key (e.g., `V1`, `V2`, `V3`, `V4`) or
- Class imbalance (76% vs. 24%) → Use evaluation metrics like AUC-ROC, F1-score, or precision-recall
- Outliers in `V2` and `V3` (IQR/z-score flags) → Validate outliers clinically (e.g., are values >Q75 + 1.5*IQR
- Lack of feature names/descriptions → Document or infer feature meanings (e.g., `V1` = donor visits, `V3` =

## Unmatched findings (unverified, not refuted)

- V1 → Apply log transformation (e.g., `log(V1 + 1)`) or bin into quantiles High zero ratio (0.67%) and right-skewed distribution (mean > median)
- V2 → Create interaction terms with `V1` (e.g., `V1 * V2`) or normalize by Both `V2` and `V3` have extreme outliers (IQR-based) and may interact
- V3 → Scale to a meaningful range (e.g., `V3 / 1000`) or flag extreme values Extreme skewness and outliers may dominate model weights; scaling or
- V4 → No transformation needed, but consider binning into clinically Normal distribution (no outliers) and moderate spread; binning could
- Class → Resample (oversample class `2` or undersample class `1`) or use Severe imbalance (76% vs. 24%) risks biased models; mitigation is critical
- Class imbalance (76% vs. 24%) → Use evaluation metrics like AUC-ROC, F1-score, or precision-recall
- Outliers in `V2` and `V3` (IQR/z-score flags) → Validate outliers clinically (e.g., are values >Q75 + 1.5*IQR
- Lack of feature names/descriptions → Document or infer feature meanings (e.g., `V1` = donor visits, `V3` =

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
