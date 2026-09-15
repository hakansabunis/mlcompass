# Scoring — exp-20260915-eb4c19-1464-llm-deepseek-flash-r3

Dataset: openml-1464 (blood-transfusion-service-center)
Configuration: llm:deepseek-flash
Status: completed
Scoring version: 2.0; ground-truth registry 1.1

Scope scored: LLM output only, from the first of ✨ Recommended models, 🔧 Feature engineering, ⚠ Pitfalls onward.

known_issues: 1
correct_detections: 1  ['DUP-1464']
missed_issues: 0  []
unverified_findings: 10  (= 5 unmatched detection claims + 5 suggestions)

A detection claim asserts a defect and is checked against the registry;
it comes from `⚠ Warnings` or `⚠ Pitfalls`. A suggestion comes from
`🔧 Feature engineering` or `✨ Recommended models` and has no ground
truth to match against, so it is counted apart rather than pooled with
claims that could have matched and did not.

## Findings in scope

- V3 → Apply log1p (or a rank/quantile transform); keep the raw value as a V3 is strongly right-skewed (mean 1378.7 vs median 1000, max 12500, std
- V2 → log1p(V2) and derive V3/V2 (average volume per donation). V2 (frequency, median 4, max 50) and V3 (total volume) are on different
- V4 → Derive V4/V2 (average months between donations) and V1/V4 (recency as V4 is months since first donation (median 28, range 2-98) while V2 counts
- V1 → Add a log1p(V1) version and a coarse bucket (e.g., 0-2, 3-6, 7-12, 13+ V1 (recency, q25=2.75, q50=7, q75=14, max 74) has a long tail and a
- V1, V2, V3 → Add pairwise products/ratios of the log-transformed variables Only 4 raw features are available; recency, frequency and monetary volume
- 215 exact duplicate rows (28.7% of 748) flagged in warnings. → De-duplicate before splitting, or use GroupKFold/grouped split on a row
- Class imbalance: Class=2 is 178/748 (23.8%) vs Class=1 at 76.2%. → Use stratified k-fold, report PR-AUC and F1 rather than plain accuracy,
- Target labels are 1 and 2, not 0 and 1. → Remap Class to {0,1} before training so sklearn metrics (roc_auc,
- Outliers: 45 IQR outliers in V2 and V3 (and 7 in V1), with max V3 = 12500 → Don't blindly trim - tree models are outlier-robust, but for
- Small dataset (748 rows, 4 features) makes a single train/test split → Use repeated stratified k-fold CV (e.g., 5x5) with a fixed seed and
- Feature scales differ by orders of magnitude (V1 ~0-74, V2 ~1-50, V3 → Standardize or log-transform features inside a pipeline (fit on training

## Unmatched detection claims (unverified, not refuted)

- Class imbalance: Class=2 is 178/748 (23.8%) vs Class=1 at 76.2%. → Use stratified k-fold, report PR-AUC and F1 rather than plain accuracy,
- Target labels are 1 and 2, not 0 and 1. → Remap Class to {0,1} before training so sklearn metrics (roc_auc,
- Outliers: 45 IQR outliers in V2 and V3 (and 7 in V1), with max V3 = 12500 → Don't blindly trim - tree models are outlier-robust, but for
- Small dataset (748 rows, 4 features) makes a single train/test split → Use repeated stratified k-fold CV (e.g., 5x5) with a fixed seed and
- Feature scales differ by orders of magnitude (V1 ~0-74, V2 ~1-50, V3 → Standardize or log-transform features inside a pipeline (fit on training

## Suggestions (no ground truth to match against)

- V3 → Apply log1p (or a rank/quantile transform); keep the raw value as a V3 is strongly right-skewed (mean 1378.7 vs median 1000, max 12500, std
- V2 → log1p(V2) and derive V3/V2 (average volume per donation). V2 (frequency, median 4, max 50) and V3 (total volume) are on different
- V4 → Derive V4/V2 (average months between donations) and V1/V4 (recency as V4 is months since first donation (median 28, range 2-98) while V2 counts
- V1 → Add a log1p(V1) version and a coarse bucket (e.g., 0-2, 3-6, 7-12, 13+ V1 (recency, q25=2.75, q50=7, q75=14, max 74) has a long tail and a
- V1, V2, V3 → Add pairwise products/ratios of the log-transformed variables Only 4 raw features are available; recency, frequency and monetary volume

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['DUP-1464']

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
