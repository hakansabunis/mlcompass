# Scoring — exp-20260915-eb4c19-1464-llm-mistral-ministral-8b-r3

Dataset: openml-1464 (blood-transfusion-service-center)
Configuration: llm:mistral-ministral-8b
Status: completed
Scoring version: 2.0; ground-truth registry 1.1

Scope scored: LLM output only, from the first of ✨ Recommended models, 🔧 Feature engineering, ⚠ Pitfalls onward.

known_issues: 1
correct_detections: 1  ['DUP-1464']
missed_issues: 0  []
unverified_findings: 8  (= 3 unmatched detection claims + 5 suggestions)

A detection claim asserts a defect and is checked against the registry;
it comes from `⚠ Warnings` or `⚠ Pitfalls`. A suggestion comes from
`🔧 Feature engineering` or `✨ Recommended models` and has no ground
truth to match against, so it is counted apart rather than pooled with
claims that could have matched and did not.

## Findings in scope

- V1 → Apply log transformation or winsorization to reduce skew and outlier High zero_ratio (0.67%) and outliers (IQR/Z-score) suggest potential data
- V2 → Bin into quantiles (e.g., low/medium/high) or apply log transformation Extreme outliers (45 IQR, 12 Z-score) and skewed distribution (min=1,
- V3 → Apply log transformation or normalize by V4 (if domain-valid) Extreme outliers (45 IQR, 12 Z-score) and wide range (250–12500) suggest
- V4 → Check for domain-specific interactions with V1/V2/V3 (e.g., ratio No outliers but moderate spread; may act as a moderator for other features
- Class → Resample (oversample class 2 or undersample class 1) or use Severe imbalance (76% vs. 24%) may bias model toward majority class
- Duplicate rows (215 exact duplicates, 28.7% of data) → De-duplicate via a unique key (e.g., hash of V1–V4) or split on a group
- Class imbalance (76% vs. 24%) → Use AUC-ROC, F1-score, or precision-recall curves instead of accuracy;
- Outliers in V2 and V3 → Validate outliers (e.g., check for data entry errors) or apply robust
- Small dataset size (748 rows) → Use cross-validation (e.g., 5-fold) and avoid overfitting by

## Unmatched detection claims (unverified, not refuted)

- Class imbalance (76% vs. 24%) → Use AUC-ROC, F1-score, or precision-recall curves instead of accuracy;
- Outliers in V2 and V3 → Validate outliers (e.g., check for data entry errors) or apply robust
- Small dataset size (748 rows) → Use cross-validation (e.g., 5-fold) and avoid overfitting by

## Suggestions (no ground truth to match against)

- V1 → Apply log transformation or winsorization to reduce skew and outlier High zero_ratio (0.67%) and outliers (IQR/Z-score) suggest potential data
- V2 → Bin into quantiles (e.g., low/medium/high) or apply log transformation Extreme outliers (45 IQR, 12 Z-score) and skewed distribution (min=1,
- V3 → Apply log transformation or normalize by V4 (if domain-valid) Extreme outliers (45 IQR, 12 Z-score) and wide range (250–12500) suggest
- V4 → Check for domain-specific interactions with V1/V2/V3 (e.g., ratio No outliers but moderate spread; may act as a moderator for other features
- Class → Resample (oversample class 2 or undersample class 1) or use Severe imbalance (76% vs. 24%) may bias model toward majority class

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['DUP-1464']

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
