# Scoring — exp-20260915-f8ba2f-1480-llm-deepseek-flash-r1

Dataset: openml-1480 (ilpd)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1480']
missed_issues: 0  []
unverified_findings: 12

## Findings reported by the tool

- 13 exact duplicate row(s) (2.2% of the data). A random split puts copies of
- V2 → Encode as binary is_male (Male=1, Female=0) or one-hot; drop the V2 has only 2 categories (441 Male, 142 Female) and no missing values.
- V3, V4, V5, V6, V7 → Apply log1p transform, then standardize for linear/SVM These columns are right-skewed with many IQR outliers (V3 84, V4 81, V5
- V3 and V4 → Add direct/total bilirubin ratio V4/(V3+epsilon) and difference Total bilirubin (V3) and direct bilirubin (V4) are clinically related;
- V6 and V7 → Add AST/ALT ratio V7/(V6+epsilon). Classic liver enzyme ratio; V6 and V7 are skewed and likely interact in
- V5 → Log-transform and optionally add ratio to V8 or V9 (e.g., V5 is right-skewed with range 63-2110 and 69 IQR outliers; alkaline
- V1 → Keep age, add clinically meaningful bins or a squared term. V1 ranges 4-90 with median 45; age-risk relationships are often nonlinear.
- V8, V9, V10 → Standardize for linear models; consider V9/(V8+epsilon) as V8 total proteins (mean 6.48), V9 albumin (mean 3.14), and V10 A/G ratio
- V10 → Verify values and consider clipping to a plausible A/G range before V10 has mean 0.947, range 0.3-2.8, and q50 exactly 0.947064, which may
- 13 exact duplicate rows (2.2%) → Confirm duplicates are genuine repeated observations; if not,
- Class imbalance (Class 1: 71.4%, Class 2: 28.6%) → Use stratified splits, class_weight='balanced' or scale_pos_weight, and
- Heavy right-skew and IQR outliers in V3-V7 → Tree models are robust; for linear/SVM use log1p and robust scaling, and
- Small sample size (583 rows, 10 features) → Use repeated cross-validation, regularize aggressively, avoid
- Gender imbalance (441 Male vs 142 Female) → Stratify by V2, evaluate performance per gender subgroup, and check

## Unmatched findings (unverified, not refuted)

- V2 → Encode as binary is_male (Male=1, Female=0) or one-hot; drop the V2 has only 2 categories (441 Male, 142 Female) and no missing values.
- V3, V4, V5, V6, V7 → Apply log1p transform, then standardize for linear/SVM These columns are right-skewed with many IQR outliers (V3 84, V4 81, V5
- V3 and V4 → Add direct/total bilirubin ratio V4/(V3+epsilon) and difference Total bilirubin (V3) and direct bilirubin (V4) are clinically related;
- V6 and V7 → Add AST/ALT ratio V7/(V6+epsilon). Classic liver enzyme ratio; V6 and V7 are skewed and likely interact in
- V5 → Log-transform and optionally add ratio to V8 or V9 (e.g., V5 is right-skewed with range 63-2110 and 69 IQR outliers; alkaline
- V1 → Keep age, add clinically meaningful bins or a squared term. V1 ranges 4-90 with median 45; age-risk relationships are often nonlinear.
- V8, V9, V10 → Standardize for linear models; consider V9/(V8+epsilon) as V8 total proteins (mean 6.48), V9 albumin (mean 3.14), and V10 A/G ratio
- V10 → Verify values and consider clipping to a plausible A/G range before V10 has mean 0.947, range 0.3-2.8, and q50 exactly 0.947064, which may
- Class imbalance (Class 1: 71.4%, Class 2: 28.6%) → Use stratified splits, class_weight='balanced' or scale_pos_weight, and
- Heavy right-skew and IQR outliers in V3-V7 → Tree models are robust; for linear/SVM use log1p and robust scaling, and
- Small sample size (583 rows, 10 features) → Use repeated cross-validation, regularize aggressively, avoid
- Gender imbalance (441 Male vs 142 Female) → Stratify by V2, evaluate performance per gender subgroup, and check

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
