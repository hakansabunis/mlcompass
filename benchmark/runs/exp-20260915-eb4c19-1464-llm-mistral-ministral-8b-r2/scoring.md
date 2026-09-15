# Scoring — exp-20260915-eb4c19-1464-llm-mistral-ministral-8b-r2

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

- V1 → Apply log transformation (e.g., `log(V1 + 1)`) or bin into quantiles High zero ratio (0.67%) and right-skewed distribution (mean > median)
- V2 → Create interaction terms with `V1` (e.g., `V1 * V2`) or normalize by Both `V2` and `V3` have extreme outliers (IQR-based) and may interact
- V3 → Scale to a meaningful range (e.g., `V3 / 1000`) or flag extreme values Extreme skewness and outliers may dominate model weights; scaling or
- V4 → No transformation needed, but consider binning into clinically Normal distribution (no outliers) and moderate spread; binning could
- Class → Resample (oversample class `2` or undersample class `1`) or use Severe imbalance (76% vs. 24%) risks biased models; mitigation is critical
- Duplicate rows (215 exact duplicates, 28.7% of data) → De-duplicate using a unique key (e.g., `V1`, `V2`, `V3`, `V4`) or
- Class imbalance (76% vs. 24%) → Use evaluation metrics like AUC-ROC, F1-score, or precision-recall
- Outliers in `V2` and `V3` (IQR/z-score flags) → Validate outliers clinically (e.g., are values >Q75 + 1.5*IQR
- Lack of feature names/descriptions → Document or infer feature meanings (e.g., `V1` = donor visits, `V3` =

## Unmatched detection claims (unverified, not refuted)

- Class imbalance (76% vs. 24%) → Use evaluation metrics like AUC-ROC, F1-score, or precision-recall
- Outliers in `V2` and `V3` (IQR/z-score flags) → Validate outliers clinically (e.g., are values >Q75 + 1.5*IQR
- Lack of feature names/descriptions → Document or infer feature meanings (e.g., `V1` = donor visits, `V3` =

## Suggestions (no ground truth to match against)

- V1 → Apply log transformation (e.g., `log(V1 + 1)`) or bin into quantiles High zero ratio (0.67%) and right-skewed distribution (mean > median)
- V2 → Create interaction terms with `V1` (e.g., `V1 * V2`) or normalize by Both `V2` and `V3` have extreme outliers (IQR-based) and may interact
- V3 → Scale to a meaningful range (e.g., `V3 / 1000`) or flag extreme values Extreme skewness and outliers may dominate model weights; scaling or
- V4 → No transformation needed, but consider binning into clinically Normal distribution (no outliers) and moderate spread; binning could
- Class → Resample (oversample class `2` or undersample class `1`) or use Severe imbalance (76% vs. 24%) risks biased models; mitigation is critical

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['DUP-1464']

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
