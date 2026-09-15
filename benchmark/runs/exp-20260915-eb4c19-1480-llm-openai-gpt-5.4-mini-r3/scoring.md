# Scoring — exp-20260915-eb4c19-1480-llm-openai-gpt-5.4-mini-r3

Dataset: openml-1480 (ilpd)
Configuration: llm:openai-gpt-5.4-mini
Status: completed
Scoring version: 2.0; ground-truth registry 1.1

Scope scored: LLM output only, from the first of ✨ Recommended models, 🔧 Feature engineering, ⚠ Pitfalls onward.

known_issues: 1
correct_detections: 1  ['DUP-1480']
missed_issues: 0  []
unverified_findings: 11  (= 5 unmatched detection claims + 6 suggestions)

A detection claim asserts a defect and is checked against the registry;
it comes from `⚠ Warnings` or `⚠ Pitfalls`. A suggestion comes from
`🔧 Feature engineering` or `✨ Recommended models` and has no ground
truth to match against, so it is counted apart rather than pooled with
claims that could have matched and did not.

## Findings in scope

- V2 → Encode as binary indicator (e.g., Male=1, Female=0) and optionally V2 has only two categories and is likely to add a small but useful
- V3, V4, V5, V6, V7 → Apply log1p or Box-Cox/Yeo-Johnson transforms; also These columns show heavy right skew and many outliers (especially V3–V7),
- V5, V6, V7 → Create ratio and difference features such as V5/V6, V5/V7, The three variables have very different ranges and many outliers, so
- V8, V9, V10 → Try polynomial or spline terms, and interactions with V3–V7; These variables are more compact and less outlier-prone, so they may serve
- V1 → Treat as a continuous age-like feature and test nonlinear bins or V1 spans 4 to 90 with a broad central mass, suggesting age effects may be
- All numeric columns → Add missingness indicators only if any missingness Current analysis shows no missing values, so the main benefit comes from
- Exact duplicate rows (13 duplicates, 2.2%) → Verify whether duplicates are legitimate repeat measurements; if not,
- Potential target leakage from random splitting with duplicates → Use a deduplicated dataset or a split strategy that prevents
- Class imbalance (Class 1 about 71%, Class 2 about 29%) → Use stratified splits, report balanced metrics such as
- Heavy skew and outliers in several predictors (especially V3–V7) → Prefer robust preprocessing, transformations, and tree-based baselines;
- Small sample size (583 rows) → Use cross-validation, keep model complexity controlled, and compare
- Possible ambiguity in column semantics (generic names V1–V10) → Map columns to domain meaning if possible, because feature engineering

## Unmatched detection claims (unverified, not refuted)

- Potential target leakage from random splitting with duplicates → Use a deduplicated dataset or a split strategy that prevents
- Class imbalance (Class 1 about 71%, Class 2 about 29%) → Use stratified splits, report balanced metrics such as
- Heavy skew and outliers in several predictors (especially V3–V7) → Prefer robust preprocessing, transformations, and tree-based baselines;
- Small sample size (583 rows) → Use cross-validation, keep model complexity controlled, and compare
- Possible ambiguity in column semantics (generic names V1–V10) → Map columns to domain meaning if possible, because feature engineering

## Suggestions (no ground truth to match against)

- V2 → Encode as binary indicator (e.g., Male=1, Female=0) and optionally V2 has only two categories and is likely to add a small but useful
- V3, V4, V5, V6, V7 → Apply log1p or Box-Cox/Yeo-Johnson transforms; also These columns show heavy right skew and many outliers (especially V3–V7),
- V5, V6, V7 → Create ratio and difference features such as V5/V6, V5/V7, The three variables have very different ranges and many outliers, so
- V8, V9, V10 → Try polynomial or spline terms, and interactions with V3–V7; These variables are more compact and less outlier-prone, so they may serve
- V1 → Treat as a continuous age-like feature and test nonlinear bins or V1 spans 4 to 90 with a broad central mass, suggesting age effects may be
- All numeric columns → Add missingness indicators only if any missingness Current analysis shows no missing values, so the main benefit comes from

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['DUP-1480']

- 13 exact duplicate row(s) (2.2% of the data). A random split puts copies of

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
