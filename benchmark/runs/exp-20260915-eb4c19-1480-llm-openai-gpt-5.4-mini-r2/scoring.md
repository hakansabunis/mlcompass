# Scoring — exp-20260915-eb4c19-1480-llm-openai-gpt-5.4-mini-r2

Dataset: openml-1480 (ilpd)
Configuration: llm:openai-gpt-5.4-mini
Status: completed
Scoring version: 2.0; ground-truth registry 1.1

Scope scored: LLM output only, from the first of ✨ Recommended models, 🔧 Feature engineering, ⚠ Pitfalls onward.

known_issues: 1
correct_detections: 1  ['DUP-1480']
missed_issues: 0  []
unverified_findings: 9  (= 3 unmatched detection claims + 6 suggestions)

A detection claim asserts a defect and is checked against the registry;
it comes from `⚠ Warnings` or `⚠ Pitfalls`. A suggestion comes from
`🔧 Feature engineering` or `✨ Recommended models` and has no ground
truth to match against, so it is counted apart rather than pooled with
claims that could have matched and did not.

## Findings in scope

- V2 → Encode as binary indicator (e.g., Male=1, Female=0) and consider a V2 has only 2 categories, so simple encoding is sufficient and
- V3, V4, V5, V6, V7 → Apply log1p or robust scaling; optionally winsorize These columns have substantial right-skew and many outliers (e.g.,
- V1 → Treat as age-like numeric feature; also consider age bins or spline V1 looks like a bounded continuous variable (4 to 90) and nonlinear age
- V8, V9, V10 → Keep as continuous predictors; try standardized versions and V8-V10 are comparatively compact numeric measures with less extreme skew
- V3 x V4, V5 x V6, V6 x V7 → Create ratio or product features where Several lab-style variables (V3-V7) vary over wide ranges;
- All numeric columns → Try missingness-free but robust preprocessing: No missing values are present, but the mix of scales and outliers makes
- Duplicate rows (13 exact duplicates, 2.2% of data) → Verify whether duplicates are genuine repeated measurements; if not,
- Class imbalance in Class (about 71% vs 29%) → Use stratified train/validation splits, consider class_weight='balanced'
- Heavy-tailed numeric features and many outliers in V3-V7 → Use robust preprocessing, consider log transforms or winsorization, and
- Potential data leakage from random splitting due to duplicates → Deduplicate before splitting or ensure duplicated records cannot appear

## Unmatched detection claims (unverified, not refuted)

- Class imbalance in Class (about 71% vs 29%) → Use stratified train/validation splits, consider class_weight='balanced'
- Heavy-tailed numeric features and many outliers in V3-V7 → Use robust preprocessing, consider log transforms or winsorization, and
- Potential data leakage from random splitting due to duplicates → Deduplicate before splitting or ensure duplicated records cannot appear

## Suggestions (no ground truth to match against)

- V2 → Encode as binary indicator (e.g., Male=1, Female=0) and consider a V2 has only 2 categories, so simple encoding is sufficient and
- V3, V4, V5, V6, V7 → Apply log1p or robust scaling; optionally winsorize These columns have substantial right-skew and many outliers (e.g.,
- V1 → Treat as age-like numeric feature; also consider age bins or spline V1 looks like a bounded continuous variable (4 to 90) and nonlinear age
- V8, V9, V10 → Keep as continuous predictors; try standardized versions and V8-V10 are comparatively compact numeric measures with less extreme skew
- V3 x V4, V5 x V6, V6 x V7 → Create ratio or product features where Several lab-style variables (V3-V7) vary over wide ranges;
- All numeric columns → Try missingness-free but robust preprocessing: No missing values are present, but the mix of scales and outliers makes

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['DUP-1480']

- 13 exact duplicate row(s) (2.2% of the data). A random split puts copies of

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
