# Scoring — exp-20260915-eb4c19-1464-llm-openai-gpt-5.4-mini-r1

Dataset: openml-1464 (blood-transfusion-service-center)
Configuration: llm:openai-gpt-5.4-mini
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

- V1 → Create binned versions (e.g., quantile bins) and consider robust V1 is numeric with noticeable outliers (IQR/z-score counts) and a wide
- V2 → Try ratio and difference features with V1 and V3, such as V3/V2, V2 is numeric with outliers and may represent a count-like measure where
- V3 → Apply log1p or other monotonic transforms, and test binned categories V3 has the largest spread and strong right-tail behavior in the summary
- V4 → Use as-is, but also test polynomial or spline-style effects if using V4 is numeric with no flagged outliers, so it may carry a smoother
- V1 + V2 + V3 + V4 → Add pairwise interaction terms, especially V1×V3 and All predictors are numeric and likely describe related blood-donation
- High duplicate rate (215 exact duplicate rows, 28.7% of the dataset) → Verify whether duplicates are genuine repeated records; if not,
- Risk of data leakage from random splitting with duplicates → Use deduplication or group-based cross-validation/splitting keyed by the
- Class imbalance (Class 1 about 76.2%, Class 2 about 23.8%) → Use stratified splits and imbalance-aware metrics such as F1, balanced
- Small dataset size after accounting for duplicates → Prefer cross-validation over a single holdout split, keep models simple
- Outliers in V1, V2, and especially V3 → Use robust preprocessing, inspect extreme values, and test models that

## Unmatched detection claims (unverified, not refuted)

- Risk of data leakage from random splitting with duplicates → Use deduplication or group-based cross-validation/splitting keyed by the
- Class imbalance (Class 1 about 76.2%, Class 2 about 23.8%) → Use stratified splits and imbalance-aware metrics such as F1, balanced
- Small dataset size after accounting for duplicates → Prefer cross-validation over a single holdout split, keep models simple
- Outliers in V1, V2, and especially V3 → Use robust preprocessing, inspect extreme values, and test models that

## Suggestions (no ground truth to match against)

- V1 → Create binned versions (e.g., quantile bins) and consider robust V1 is numeric with noticeable outliers (IQR/z-score counts) and a wide
- V2 → Try ratio and difference features with V1 and V3, such as V3/V2, V2 is numeric with outliers and may represent a count-like measure where
- V3 → Apply log1p or other monotonic transforms, and test binned categories V3 has the largest spread and strong right-tail behavior in the summary
- V4 → Use as-is, but also test polynomial or spline-style effects if using V4 is numeric with no flagged outliers, so it may carry a smoother
- V1 + V2 + V3 + V4 → Add pairwise interaction terms, especially V1×V3 and All predictors are numeric and likely describe related blood-donation

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['DUP-1464']

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
