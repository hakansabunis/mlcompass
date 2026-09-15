# Scoring — exp-20260915-eb4c19-1464-llm-deepseek-flash-r1

Dataset: openml-1464 (blood-transfusion-service-center)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1464']
missed_issues: 0  []
unverified_findings: 11

## Findings reported by the tool

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies
- V3 → log1p(V3) or a rank/quantile transform; optionally cap at the 99th V3 is heavily right-skewed (mean 1378.7 vs median 1000, max 12500, std
- V2 → log1p(V2) and/or winsorize the top values V2 is right-skewed (mean 5.5, median 4, max 50) with 45 IQR outliers and
- V1 → Keep as-is; optionally bin into recency buckets (e.g., <3, 3–12, >12 V1 (recency) has only 7 outliers and zero 0.7% of the time, but the
- V2, V4 → V2 / max(V4, 1) — donation frequency per month of observed tenure V2 is total donations and V4 is months since first donation; their ratio
- V3, V2 → V3 / V2 — average blood volume per donation V3 (total ccs, median 1000) divided by V2 (median 4 donations) yields a
- V1, V4 → V1 / max(V4, 1) and V4 - V1 (recency relative to tenure / active V4 (months since first donation, median 28) and V1 (months since last
- 215 exact duplicate rows (28.7% of data); a random split leaks identical → Verify whether the duplicates are genuine repeat observations; if not,
- Moderate class imbalance (Class 1 = 76.2%, Class 2 = 23.8%) → Use stratified CV and report AUC/PR-AUC plus per-class F1 rather than
- Small sample (748 rows, 4 predictors); a single random split will have high → Use repeated stratified k-fold (k=5 or 10) and report mean ± std across
- Heavy outliers/skew in V2 and V3 (45 IQR outliers each, V3 max 12.5x its → Winsorize or log1p-transform these columns, or rely on scale-invariant
- Feature names V1–V4 are opaque, so it is easy to apply the wrong transform → Map them to their documented meaning (recency, frequency, monetary/ccs,
- No missing values today, but the pipeline may silently coerce them later → Add explicit imputation/validation steps and assert dtypes (all features

## Unmatched findings (unverified, not refuted)

- V3 → log1p(V3) or a rank/quantile transform; optionally cap at the 99th V3 is heavily right-skewed (mean 1378.7 vs median 1000, max 12500, std
- V2 → log1p(V2) and/or winsorize the top values V2 is right-skewed (mean 5.5, median 4, max 50) with 45 IQR outliers and
- V1 → Keep as-is; optionally bin into recency buckets (e.g., <3, 3–12, >12 V1 (recency) has only 7 outliers and zero 0.7% of the time, but the
- V2, V4 → V2 / max(V4, 1) — donation frequency per month of observed tenure V2 is total donations and V4 is months since first donation; their ratio
- V3, V2 → V3 / V2 — average blood volume per donation V3 (total ccs, median 1000) divided by V2 (median 4 donations) yields a
- V1, V4 → V1 / max(V4, 1) and V4 - V1 (recency relative to tenure / active V4 (months since first donation, median 28) and V1 (months since last
- Moderate class imbalance (Class 1 = 76.2%, Class 2 = 23.8%) → Use stratified CV and report AUC/PR-AUC plus per-class F1 rather than
- Small sample (748 rows, 4 predictors); a single random split will have high → Use repeated stratified k-fold (k=5 or 10) and report mean ± std across
- Heavy outliers/skew in V2 and V3 (45 IQR outliers each, V3 max 12.5x its → Winsorize or log1p-transform these columns, or rely on scale-invariant
- Feature names V1–V4 are opaque, so it is easy to apply the wrong transform → Map them to their documented meaning (recency, frequency, monetary/ccs,
- No missing values today, but the pipeline may silently coerce them later → Add explicit imputation/validation steps and assert dtypes (all features

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
