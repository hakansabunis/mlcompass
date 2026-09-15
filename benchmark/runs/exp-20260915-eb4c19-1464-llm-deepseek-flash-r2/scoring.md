# Scoring — exp-20260915-eb4c19-1464-llm-deepseek-flash-r2

Dataset: openml-1464 (blood-transfusion-service-center)
Configuration: llm:deepseek-flash
Status: completed
Scoring version: 2.0; ground-truth registry 1.1

Scope scored: LLM output only, from the first of ✨ Recommended models, 🔧 Feature engineering, ⚠ Pitfalls onward.

known_issues: 1
correct_detections: 1  ['DUP-1464']
missed_issues: 0  []
unverified_findings: 9  (= 4 unmatched detection claims + 5 suggestions)

A detection claim asserts a defect and is checked against the registry;
it comes from `⚠ Warnings` or `⚠ Pitfalls`. A suggestion comes from
`🔧 Feature engineering` or `✨ Recommended models` and has no ground
truth to match against, so it is counted apart rather than pooled with
claims that could have matched and did not.

## Findings in scope

- V3 → log1p transform and/or robust (median/IQR) scaling of V3 V3 is strongly right-skewed (mean 1378.7 vs median 1000, max 12500, 45 IQR
- V2 → derive donations-per-month = V2 / max(V4,1) and V3-per-donation = V3 / V2 (frequency, mean 5.5) and V4 (months since first donation, mean 34.3)
- V1 → derive recency_ratio = V1 / max(V4,1) and recency_rank vs V1 quantiles V1 (months since last donation, mean 9.5, max 74) should be interpreted
- V2 → bin V1/V2/V4 into quantile buckets (e.g. qcut into 4-5 bins) as extra All four predictors are small-range integers with skewed counts; binning
- V4 → add polynomial/interaction terms V1*V2, V1*V4, V2*V3 for the linear With only four raw features the logistic model has little capacity;
- 215 exact duplicate rows (28.7% of the data) reported in warnings → Confirm duplicates are genuine repeat observations; if they are
- Class imbalance: Class 1 is 76.2% vs Class 2 at 23.8% → Report precision/recall, F1, and ROC-AUC or PR-AUC instead of raw
- Very small dataset (748 rows, 4 features) makes any single train/test split → Use stratified k-fold cross-validation repeated with several seeds, and
- Heavy-tailed predictors with flagged outliers (V2: 45 IQR outliers; V3: 45 → Prefer tree models or robust transforms over raw-scale linear models; if
- Column names V1-V4 are opaque, and the Class encoding is 1/2 rather than → Map Class to a 0/1 binary label explicitly before training and verify

## Unmatched detection claims (unverified, not refuted)

- Class imbalance: Class 1 is 76.2% vs Class 2 at 23.8% → Report precision/recall, F1, and ROC-AUC or PR-AUC instead of raw
- Very small dataset (748 rows, 4 features) makes any single train/test split → Use stratified k-fold cross-validation repeated with several seeds, and
- Heavy-tailed predictors with flagged outliers (V2: 45 IQR outliers; V3: 45 → Prefer tree models or robust transforms over raw-scale linear models; if
- Column names V1-V4 are opaque, and the Class encoding is 1/2 rather than → Map Class to a 0/1 binary label explicitly before training and verify

## Suggestions (no ground truth to match against)

- V3 → log1p transform and/or robust (median/IQR) scaling of V3 V3 is strongly right-skewed (mean 1378.7 vs median 1000, max 12500, 45 IQR
- V2 → derive donations-per-month = V2 / max(V4,1) and V3-per-donation = V3 / V2 (frequency, mean 5.5) and V4 (months since first donation, mean 34.3)
- V1 → derive recency_ratio = V1 / max(V4,1) and recency_rank vs V1 quantiles V1 (months since last donation, mean 9.5, max 74) should be interpreted
- V2 → bin V1/V2/V4 into quantile buckets (e.g. qcut into 4-5 bins) as extra All four predictors are small-range integers with skewed counts; binning
- V4 → add polynomial/interaction terms V1*V2, V1*V4, V2*V3 for the linear With only four raw features the logistic model has little capacity;

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['DUP-1464']

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
