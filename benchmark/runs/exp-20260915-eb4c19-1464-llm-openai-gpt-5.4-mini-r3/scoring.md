# Scoring — exp-20260915-eb4c19-1464-llm-openai-gpt-5.4-mini-r3

Dataset: openml-1464 (blood-transfusion-service-center)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1464']
missed_issues: 0  []
unverified_findings: 10

## Findings reported by the tool

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies
- V1 → Try robust scaling and bucketed versions (e.g., quantile bins) of V1 V1 is numeric with a right tail and a few outliers (min 0, max 74,
- V2 → Consider log1p transform or robust scaling for V2 V2 shows clear outliers and a skewed spread (outliers flagged, max 50 vs
- V3 → Consider log1p transform and/or winsorization for V3 V3 has the strongest scale and outlier pattern (250 to 12500, many
- V4 → Use V4 as-is, but also test simple threshold/binned features V4 is numeric with no outliers flagged and a relatively compact range (2
- V1, V2, V3, V4 → Add cross-feature interactions such as V1/V2, V3/V2, All predictors are numeric and likely encode related blood donation
- V1, V2, V3 → Create ratio or recency-style composites if domain semantics The variables appear to be related donation-count/volume/time features,
- Exact duplicate rows are common (215 duplicates, 28.7% of the data) → Verify whether duplicates are genuine repeated observations; if not,
- Potential train-test leakage from random splitting with duplicates → Perform deduplication or split on a stable entity/group key if
- Class imbalance in the target (Class 1 about 76%, Class 2 about 24%) → Use stratified splits, class weights or balanced loss, and evaluate with
- Small dataset size (748 rows) increases variance and overfitting risk → Prefer simple baselines first, use cross-validation, limit model
- Several numeric features have strong outliers and skew (especially V2 and → Use robust preprocessing, inspect whether extreme values are valid, and

## Unmatched findings (unverified, not refuted)

- V1 → Try robust scaling and bucketed versions (e.g., quantile bins) of V1 V1 is numeric with a right tail and a few outliers (min 0, max 74,
- V2 → Consider log1p transform or robust scaling for V2 V2 shows clear outliers and a skewed spread (outliers flagged, max 50 vs
- V3 → Consider log1p transform and/or winsorization for V3 V3 has the strongest scale and outlier pattern (250 to 12500, many
- V4 → Use V4 as-is, but also test simple threshold/binned features V4 is numeric with no outliers flagged and a relatively compact range (2
- V1, V2, V3, V4 → Add cross-feature interactions such as V1/V2, V3/V2, All predictors are numeric and likely encode related blood donation
- V1, V2, V3 → Create ratio or recency-style composites if domain semantics The variables appear to be related donation-count/volume/time features,
- Potential train-test leakage from random splitting with duplicates → Perform deduplication or split on a stable entity/group key if
- Class imbalance in the target (Class 1 about 76%, Class 2 about 24%) → Use stratified splits, class weights or balanced loss, and evaluate with
- Small dataset size (748 rows) increases variance and overfitting risk → Prefer simple baselines first, use cross-validation, limit model
- Several numeric features have strong outliers and skew (especially V2 and → Use robust preprocessing, inspect whether extreme values are valid, and

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
