# Scoring — exp-20260915-eb4c19-1480-llm-mistral-ministral-8b-r2

Dataset: openml-1480 (ilpd)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1480']
missed_issues: 0  []
unverified_findings: 9

## Findings reported by the tool

- 13 exact duplicate row(s) (2.2% of the data). A random split puts copies of
- V1 → Standardize or normalize (e.g., MinMaxScaler) given its wide range Mean (44.7) and std (16.2) suggest a skewed distribution; scaling may
- V3 → Apply log transformation or Winsorization to mitigate outliers (84 High outlier count (17%) may distort model estimates; log(V3+1) or capping
- V4 → Bin into quartiles or use target encoding if cardinality is low in Sparse but skewed (mean=1.48, std=2.81); binning may reveal patterns or
- V5 → Create interaction terms with V6/V7 (e.g., V5*V6) or log-transform Both V5 (IQR=122.5) and V6 (IQR=37.5) show extreme skew; interactions may
- V2 → One-hot encode (or embed if using neural nets) and check for Gender imbalance (441:142) may correlate with other features; encoding
- Class → Resample (SMOTE/ADASYN) or use class weights (e.g., Class imbalance (71%:29%) may bias models toward majority class;
- Duplicate rows (13 exact copies, 2.2% of data) → De-duplicate via `df.drop_duplicates()` before splitting, or split on a
- Outliers in V3–V7 (IQR/z-score counts >10%) → Validate outliers via domain knowledge; if noise, cap/trim; if signal,
- No train-test split validation (warnings imply manual split) → Use `train_test_split` with `stratify=Class` to preserve class balance;
- No feature-target correlation analysis → Check pairwise correlations (e.g., `pd.crosstab(V2, Class)`) and target

## Unmatched findings (unverified, not refuted)

- V1 → Standardize or normalize (e.g., MinMaxScaler) given its wide range Mean (44.7) and std (16.2) suggest a skewed distribution; scaling may
- V3 → Apply log transformation or Winsorization to mitigate outliers (84 High outlier count (17%) may distort model estimates; log(V3+1) or capping
- V4 → Bin into quartiles or use target encoding if cardinality is low in Sparse but skewed (mean=1.48, std=2.81); binning may reveal patterns or
- V5 → Create interaction terms with V6/V7 (e.g., V5*V6) or log-transform Both V5 (IQR=122.5) and V6 (IQR=37.5) show extreme skew; interactions may
- V2 → One-hot encode (or embed if using neural nets) and check for Gender imbalance (441:142) may correlate with other features; encoding
- Class → Resample (SMOTE/ADASYN) or use class weights (e.g., Class imbalance (71%:29%) may bias models toward majority class;
- Outliers in V3–V7 (IQR/z-score counts >10%) → Validate outliers via domain knowledge; if noise, cap/trim; if signal,
- No train-test split validation (warnings imply manual split) → Use `train_test_split` with `stratify=Class` to preserve class balance;
- No feature-target correlation analysis → Check pairwise correlations (e.g., `pd.crosstab(V2, Class)`) and target

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
