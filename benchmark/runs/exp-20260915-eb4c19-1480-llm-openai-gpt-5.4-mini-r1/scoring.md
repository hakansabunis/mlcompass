# Scoring — exp-20260915-eb4c19-1480-llm-openai-gpt-5.4-mini-r1

Dataset: openml-1480 (ilpd)
Configuration: llm:openai-gpt-5.4-mini
Status: completed
Scoring version: 2.0; ground-truth registry 1.1

Scope scored: LLM output only, from the first of ✨ Recommended models, 🔧 Feature engineering, ⚠ Pitfalls onward.

known_issues: 1
correct_detections: 1  ['DUP-1480']
missed_issues: 0  []
unverified_findings: 12  (= 3 unmatched detection claims + 9 suggestions)

A detection claim asserts a defect and is checked against the registry;
it comes from `⚠ Warnings` or `⚠ Pitfalls`. A suggestion comes from
`🔧 Feature engineering` or `✨ Recommended models` and has no ground
truth to match against, so it is counted apart rather than pooled with
claims that could have matched and did not.

## Findings in scope

- V2 → One-hot encode the Male/Female category; optionally add a binary V2 is the only categorical predictor and has low cardinality, so explicit
- V3 → Apply a log1p transform or robust scaling; consider winsorizing V3 is heavily right-skewed with many outliers (IQR count 84, z-score count
- V4 → Apply a log1p/robust scaling transform and consider clipping extreme V4 shows strong skew and many outliers (IQR count 81, z-score count 19),
- V5 → Use log1p scaling and create a ratio feature with V6 or V7 if domain V5 has a wide range with strong right tail (max 2110 vs median 208) and
- V6 → Use log1p scaling; consider interaction terms with V5 and V7. V6 is also highly skewed with outliers (IQR count 73) and may carry more
- V7 → Use log1p scaling; add cross-features such as V7/V5 or V7/V6 if they V7 has the largest extreme values among the numeric features (max 4929)
- V1 → Keep as-is or standardize; if treating it as age, consider binning V1 is a stable numeric feature with no reported outliers and likely
- V8 → Standardize/robust-scale; check whether nonlinearity with the target V8 is relatively compact but still numeric and may interact with the other
- V10 → Standardize/robust-scale and consider interaction terms with the V10 is a low-variance numeric feature; small shifts may matter, especially
- Duplicate rows present (13 exact duplicates, 2.2% of data) → Verify whether duplicates are true repeated records; if not, remove them
- Class imbalance in the target (Class 1: 71.4%, Class 2: 28.6%) → Use stratified splits and metrics suited to imbalance (e.g., F1,
- Strong outliers and skew in several numeric predictors (especially V3, V4, → Prefer robust preprocessing (log transforms, winsorization, robust
- Small sample size for a binary classification problem (583 rows) → Use cross-validation rather than a single train/test split, keep models

## Unmatched detection claims (unverified, not refuted)

- Class imbalance in the target (Class 1: 71.4%, Class 2: 28.6%) → Use stratified splits and metrics suited to imbalance (e.g., F1,
- Strong outliers and skew in several numeric predictors (especially V3, V4, → Prefer robust preprocessing (log transforms, winsorization, robust
- Small sample size for a binary classification problem (583 rows) → Use cross-validation rather than a single train/test split, keep models

## Suggestions (no ground truth to match against)

- V2 → One-hot encode the Male/Female category; optionally add a binary V2 is the only categorical predictor and has low cardinality, so explicit
- V3 → Apply a log1p transform or robust scaling; consider winsorizing V3 is heavily right-skewed with many outliers (IQR count 84, z-score count
- V4 → Apply a log1p/robust scaling transform and consider clipping extreme V4 shows strong skew and many outliers (IQR count 81, z-score count 19),
- V5 → Use log1p scaling and create a ratio feature with V6 or V7 if domain V5 has a wide range with strong right tail (max 2110 vs median 208) and
- V6 → Use log1p scaling; consider interaction terms with V5 and V7. V6 is also highly skewed with outliers (IQR count 73) and may carry more
- V7 → Use log1p scaling; add cross-features such as V7/V5 or V7/V6 if they V7 has the largest extreme values among the numeric features (max 4929)
- V1 → Keep as-is or standardize; if treating it as age, consider binning V1 is a stable numeric feature with no reported outliers and likely
- V8 → Standardize/robust-scale; check whether nonlinearity with the target V8 is relatively compact but still numeric and may interact with the other
- V10 → Standardize/robust-scale and consider interaction terms with the V10 is a low-variance numeric feature; small shifts may matter, especially

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['DUP-1480']

- 13 exact duplicate row(s) (2.2% of the data). A random split puts copies of

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
