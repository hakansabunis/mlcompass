# Scoring — exp-20260915-eb4c19-1480-llm-deepseek-flash-r2

Dataset: openml-1480 (ilpd)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1480']
missed_issues: 0  []
unverified_findings: 12

## Findings reported by the tool

- 13 exact duplicate row(s) (2.2% of the data). A random split puts copies of
- V3, V4, V5, V6, V7 → Apply log1p (or Box-Cox) and keep the untransformed Each has std far exceeding the median (V3 mean 3.30 vs median 1.0; V7 mean
- V7 and V6 → Add the De Ritis ratio V7/V6 and the absolute difference V7-V6. Both are aminotransferases on the same scale (V6 max 2000, V7 max 4929);
- V4 and V3 → Add the direct/total bilirubin fraction V4/V3. V4 (direct) ranges only 0.1-19.7 while V3 (total) reaches 75, so the
- V8 and V9 → Derive globulin as V8 - V9 and a V9/V8 albumin fraction; V8 is total protein and V9 is albumin, so they are mechanically related
- V1 → Keep the raw age and add bins (e.g. <18, 18-40, 40-65, 65+) or a Age spans 4-90 with q25=33 and q75=58, and the min of 4 indicates a
- V2 → Encode Male/Female as a single 0/1 flag (drop-first) rather than Cardinality is 2 with 441 Male vs 142 Female, so a redundant dummy column
- V3, V4, V6, V7 → Create simple threshold indicator features (e.g. above All four outlier counts cluster at 66-84 IQR violations, consistent with a
- 13 exact duplicate rows (2.2% of data) flagged in warnings. → Confirm they are genuine repeats; if not, de-duplicate before splitting
- Mild-to-moderate class imbalance (Class 1 = 71.4%, Class 2 = 28.6%). → Report stratified precision/recall, F1 and ROC-AUC instead of accuracy;
- Label polarity of Class (1 = 416, 2 = 167) is ambiguous and both values are → Explicitly declare which class is the positive/disease class before
- Heavy right-skew and outliers in V3-V7 (e.g. V3 reaches 75 vs median 1.0; → Log-transform or winsorize for linear/kernel models; trees need no
- Collinearity among V8, V9 and V10 (total protein, albumin, derived A/G → Check VIF before interpreting logistic coefficients; for the linear
- Small sample size (583 rows, 10 features) with a skewed gender split (441 M → Use repeated stratified k-fold cross-validation and report variance

## Unmatched findings (unverified, not refuted)

- V3, V4, V5, V6, V7 → Apply log1p (or Box-Cox) and keep the untransformed Each has std far exceeding the median (V3 mean 3.30 vs median 1.0; V7 mean
- V7 and V6 → Add the De Ritis ratio V7/V6 and the absolute difference V7-V6. Both are aminotransferases on the same scale (V6 max 2000, V7 max 4929);
- V4 and V3 → Add the direct/total bilirubin fraction V4/V3. V4 (direct) ranges only 0.1-19.7 while V3 (total) reaches 75, so the
- V8 and V9 → Derive globulin as V8 - V9 and a V9/V8 albumin fraction; V8 is total protein and V9 is albumin, so they are mechanically related
- V1 → Keep the raw age and add bins (e.g. <18, 18-40, 40-65, 65+) or a Age spans 4-90 with q25=33 and q75=58, and the min of 4 indicates a
- V2 → Encode Male/Female as a single 0/1 flag (drop-first) rather than Cardinality is 2 with 441 Male vs 142 Female, so a redundant dummy column
- V3, V4, V6, V7 → Create simple threshold indicator features (e.g. above All four outlier counts cluster at 66-84 IQR violations, consistent with a
- Mild-to-moderate class imbalance (Class 1 = 71.4%, Class 2 = 28.6%). → Report stratified precision/recall, F1 and ROC-AUC instead of accuracy;
- Label polarity of Class (1 = 416, 2 = 167) is ambiguous and both values are → Explicitly declare which class is the positive/disease class before
- Heavy right-skew and outliers in V3-V7 (e.g. V3 reaches 75 vs median 1.0; → Log-transform or winsorize for linear/kernel models; trees need no
- Collinearity among V8, V9 and V10 (total protein, albumin, derived A/G → Check VIF before interpreting logistic coefficients; for the linear
- Small sample size (583 rows, 10 features) with a skewed gender split (441 M → Use repeated stratified k-fold cross-validation and report variance

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
