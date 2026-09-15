# Scoring — exp-20260915-eb4c19-1480-llm-deepseek-flash-r1

Dataset: openml-1480 (ilpd)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1480']
missed_issues: 0  []
unverified_findings: 13

## Findings reported by the tool

- 13 exact duplicate row(s) (2.2% of the data). A random split puts copies of
- V3 → Apply log1p transform; also compute indirect bilirubin = V3 - V4 V3 (total bilirubin) is strongly right-skewed (mean 3.30 vs median 1.0,
- V4 → Apply log1p transform and consider V4/V3 as a fraction V4 (direct bilirubin) shows the same skew (median 0.3, max 19.7, 81 IQR
- V7 → Compute AST/ALT (De Ritis) ratio V7/V6 and log-transform both raw V6 and V7 are the two most outlier-laden columns (73 and 66 IQR outliers,
- V5 → Apply log1p transform (or rank/quantile transform); avoid raw values V5 has range 63-2110 with std 243 versus median 208 and 69 IQR outliers,
- V8 → Derive globulin = V8 - V9 and keep the existing V10 ratio, checking V8 (total protein) is tight (std 1.09, only 8 IQR outliers) while V9
- V2 → Binary-encode Male/Female (0/1) rather than one-hot V2 has cardinality 2 with a 441/142 split, so a single indicator captures
- V1 → Keep raw age plus optional bins (e.g. <30, 30-50, 50-70, 70+) V1 is the only well-behaved column (no IQR or z-score outliers, median 45,
- 13 exact duplicate rows (2.2% of data) → Confirm they are genuine repeat observations; if not, de-duplicate
- Moderate class imbalance (Class 1 = 71.4%, Class 2 = 28.6%) → Use stratified k-fold and report AUC / PR-AUC / recall for the minority
- Severe skew and many flagged outliers in V3-V7 (66-84 IQR outliers each, → Do not blindly clip: these extremes are plausibly real clinical values
- Wildly different feature scales (V5 up to 2110 vs V10 max 2.8) → Standardize or robust-scale features for the logistic baseline and any
- Small sample (583 rows, 10 predictors) makes a single train/test split → Prefer repeated stratified k-fold cross-validation and report the spread
- Target is coded 1/2 rather than 0/1 → Remap Class to 0/1 before training so it is not accidentally treated as
- Potential definitional redundancy among V8, V9, V10 (albumin, total → Check pairwise correlation before training; if V10 is a deterministic

## Unmatched findings (unverified, not refuted)

- V3 → Apply log1p transform; also compute indirect bilirubin = V3 - V4 V3 (total bilirubin) is strongly right-skewed (mean 3.30 vs median 1.0,
- V4 → Apply log1p transform and consider V4/V3 as a fraction V4 (direct bilirubin) shows the same skew (median 0.3, max 19.7, 81 IQR
- V7 → Compute AST/ALT (De Ritis) ratio V7/V6 and log-transform both raw V6 and V7 are the two most outlier-laden columns (73 and 66 IQR outliers,
- V5 → Apply log1p transform (or rank/quantile transform); avoid raw values V5 has range 63-2110 with std 243 versus median 208 and 69 IQR outliers,
- V8 → Derive globulin = V8 - V9 and keep the existing V10 ratio, checking V8 (total protein) is tight (std 1.09, only 8 IQR outliers) while V9
- V2 → Binary-encode Male/Female (0/1) rather than one-hot V2 has cardinality 2 with a 441/142 split, so a single indicator captures
- V1 → Keep raw age plus optional bins (e.g. <30, 30-50, 50-70, 70+) V1 is the only well-behaved column (no IQR or z-score outliers, median 45,
- Moderate class imbalance (Class 1 = 71.4%, Class 2 = 28.6%) → Use stratified k-fold and report AUC / PR-AUC / recall for the minority
- Severe skew and many flagged outliers in V3-V7 (66-84 IQR outliers each, → Do not blindly clip: these extremes are plausibly real clinical values
- Wildly different feature scales (V5 up to 2110 vs V10 max 2.8) → Standardize or robust-scale features for the logistic baseline and any
- Small sample (583 rows, 10 predictors) makes a single train/test split → Prefer repeated stratified k-fold cross-validation and report the spread
- Target is coded 1/2 rather than 0/1 → Remap Class to 0/1 before training so it is not accidentally treated as
- Potential definitional redundancy among V8, V9, V10 (albumin, total → Check pairwise correlation before training; if V10 is a deterministic

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
