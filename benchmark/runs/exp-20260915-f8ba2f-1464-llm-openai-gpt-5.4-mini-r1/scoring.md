# Scoring — exp-20260915-f8ba2f-1464-llm-openai-gpt-5.4-mini-r1

Dataset: openml-1464 (blood-transfusion-service-center)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1464']
missed_issues: 0  []
unverified_findings: 8

## Findings reported by the tool

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies
- V1, V2, V3, V4 → Standardize or robust-scale the numeric inputs for linear All predictors are numeric with different scales, especially V3 versus
- V1, V2, V3 → Try log1p or winsorized variants for the right-skewed / V1, V2, and V3 show notable outliers and wide spreads in the summary
- V1, V2 → Create interaction features such as V1/V2, V1*V2, or In blood-donation style tabular data, recency and frequency-type variables
- V3, V4 → Add interaction terms or binning for V3 and V4 to capture V3 has a wide range with outliers, while V4 is more concentrated; combined
- Duplicate rows are common (215 exact duplicates, 28.7% of the data). → Verify whether duplicates are genuine repeated observations; if not,
- Potential train/test leakage from random splitting with duplicates. → Use group-aware splitting or deduplicate first; at minimum, check that
- Class imbalance in Class (about 76% vs 24%). → Use imbalance-aware metrics and training settings such as class weights,
- Small dataset size with only 748 rows. → Prefer cross-validation and simpler models first; keep the model search
- Outliers in V1, V2, and especially V3 may overly influence some models. → Inspect extreme values, consider robust scaling or transformation, and

## Unmatched findings (unverified, not refuted)

- V1, V2, V3, V4 → Standardize or robust-scale the numeric inputs for linear All predictors are numeric with different scales, especially V3 versus
- V1, V2, V3 → Try log1p or winsorized variants for the right-skewed / V1, V2, and V3 show notable outliers and wide spreads in the summary
- V1, V2 → Create interaction features such as V1/V2, V1*V2, or In blood-donation style tabular data, recency and frequency-type variables
- V3, V4 → Add interaction terms or binning for V3 and V4 to capture V3 has a wide range with outliers, while V4 is more concentrated; combined
- Potential train/test leakage from random splitting with duplicates. → Use group-aware splitting or deduplicate first; at minimum, check that
- Class imbalance in Class (about 76% vs 24%). → Use imbalance-aware metrics and training settings such as class weights,
- Small dataset size with only 748 rows. → Prefer cross-validation and simpler models first; keep the model search
- Outliers in V1, V2, and especially V3 may overly influence some models. → Inspect extreme values, consider robust scaling or transformation, and

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
