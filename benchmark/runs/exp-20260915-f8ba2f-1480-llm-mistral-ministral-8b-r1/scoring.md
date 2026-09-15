# Scoring — exp-20260915-f8ba2f-1480-llm-mistral-ministral-8b-r1

Dataset: openml-1480 (ilpd)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1480']
missed_issues: 0  []
unverified_findings: 8

## Findings reported by the tool

- 13 exact duplicate row(s) (2.2% of the data). A random split puts copies of
- V1 → Standardize or normalize (e.g., MinMaxScaler) to mitigate scale V1’s range (4–90) overlaps with V5/V6’s lower quartiles (175.5–23), but
- ['V3', 'V4'] → Bin outliers (e.g., IQR-based) into separate categories or V3 and V4 show 84/81 IQR outliers and high zero-ratio-adjacent medians
- ['V5', 'V6', 'V7'] → Derive ratios (e.g., V5/V6) or interaction terms V5/V6/V7 exhibit extreme skewness (IQR counts: 69/73/66) and wide ranges
- V2 → Encode as one-hot (e.g., `is_Male`) and check for interaction effects Gender imbalance (441:142) may interact with numeric predictors (e.g.,
- ['V8', 'V9', 'V10'] → Group into quantiles (e.g., 4 bins) to reduce noise; These features have narrow ranges (std < 1.1) but may encode subtle
- Duplicate rows (13 exact copies, 2.2% of data) → Remove duplicates before splitting or use a group key (e.g., `V1+V2`) to
- Class imbalance (28.6% minority class) → Use `class_weight='balanced'` in scikit-learn or focal loss in XGBoost;
- Outliers in V3–V7 (IQR/z-score counts: 6–84) → Robust scalers (e.g., `RobustScaler`) or outlier caps (e.g., 99th
- Lack of temporal/sequential context → If `V1`–`V10` represent time-series features (e.g., lab values),

## Unmatched findings (unverified, not refuted)

- V1 → Standardize or normalize (e.g., MinMaxScaler) to mitigate scale V1’s range (4–90) overlaps with V5/V6’s lower quartiles (175.5–23), but
- ['V3', 'V4'] → Bin outliers (e.g., IQR-based) into separate categories or V3 and V4 show 84/81 IQR outliers and high zero-ratio-adjacent medians
- ['V5', 'V6', 'V7'] → Derive ratios (e.g., V5/V6) or interaction terms V5/V6/V7 exhibit extreme skewness (IQR counts: 69/73/66) and wide ranges
- V2 → Encode as one-hot (e.g., `is_Male`) and check for interaction effects Gender imbalance (441:142) may interact with numeric predictors (e.g.,
- ['V8', 'V9', 'V10'] → Group into quantiles (e.g., 4 bins) to reduce noise; These features have narrow ranges (std < 1.1) but may encode subtle
- Class imbalance (28.6% minority class) → Use `class_weight='balanced'` in scikit-learn or focal loss in XGBoost;
- Outliers in V3–V7 (IQR/z-score counts: 6–84) → Robust scalers (e.g., `RobustScaler`) or outlier caps (e.g., 99th
- Lack of temporal/sequential context → If `V1`–`V10` represent time-series features (e.g., lab values),

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
