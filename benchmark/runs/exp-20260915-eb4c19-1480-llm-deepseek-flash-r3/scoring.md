# Scoring — exp-20260915-eb4c19-1480-llm-deepseek-flash-r3

Dataset: openml-1480 (ilpd)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1480']
missed_issues: 0  []
unverified_findings: 11

## Findings reported by the tool

- 13 exact duplicate row(s) (2.2% of the data). A random split puts copies of
- V3, V4 → log1p-transform both, then add direct/total bilirubin ratio = V4 / V3 (q50=1.0, max=75.0, 84 IQR outliers) and V4 (q50=0.3, max=19.7, 81 IQR
- V6, V7 → log1p-transform, then add AST/ALT (De Ritis) ratio = V7 / V6 and V6 (max=2000, 73 IQR outliers) and V7 (max=4929, 66 IQR outliers) are the
- V8, V9, V10 → derive globulin = V8 - V9 and check it against V10 (= V9 / V8 (total protein), V9 (albumin) and V10 (A/G ratio) are algebraically
- V5 → log1p-transform and/or quantile-bin into terciles; optionally combine V5 has mean=290.6 vs median=208.0 and 69 IQR outliers, so the raw scale is
- V1 → keep raw and add an age bin (e.g. <30 / 30-60 / >60); optionally V1 spans 4–90 with a symmetric spread (q25=33, q75=58) and no outliers, so
- V2 → binary-encode (0/1) rather than one-hot; optionally interact with V1 V2 has cardinality 2 with 441 Male / 142 Female, so a single indicator is
- 13 exact duplicate rows (2.2% of the data, flagged in warnings) → De-duplicate before splitting, or use GroupKFold with the duplicate key
- Severe right skew and heavy outlier counts in V3, V4, V5, V6, V7 (66–84 IQR → Do not apply plain standardization and then trust a linear model; use
- Class imbalance: 416 (71.4%) vs 167 (28.6%) in the Class column → Use stratified train/test and stratified CV folds,
- Ambiguous target polarity — Class is coded 1/2 rather than 0/1 → Explicitly set the positive class (e.g. map 1='liver patient') before
- Very small sample (583 rows, 10 predictors) plus sex imbalance (441 Male / → Use repeated stratified k-fold CV (e.g. 5x5) with reported
- Columns are anonymized (V1–V10), so derived-feature semantics depend on the → Confirm the column-to-biomarker mapping and units with the data owner

## Unmatched findings (unverified, not refuted)

- V3, V4 → log1p-transform both, then add direct/total bilirubin ratio = V4 / V3 (q50=1.0, max=75.0, 84 IQR outliers) and V4 (q50=0.3, max=19.7, 81 IQR
- V6, V7 → log1p-transform, then add AST/ALT (De Ritis) ratio = V7 / V6 and V6 (max=2000, 73 IQR outliers) and V7 (max=4929, 66 IQR outliers) are the
- V8, V9, V10 → derive globulin = V8 - V9 and check it against V10 (= V9 / V8 (total protein), V9 (albumin) and V10 (A/G ratio) are algebraically
- V5 → log1p-transform and/or quantile-bin into terciles; optionally combine V5 has mean=290.6 vs median=208.0 and 69 IQR outliers, so the raw scale is
- V1 → keep raw and add an age bin (e.g. <30 / 30-60 / >60); optionally V1 spans 4–90 with a symmetric spread (q25=33, q75=58) and no outliers, so
- V2 → binary-encode (0/1) rather than one-hot; optionally interact with V1 V2 has cardinality 2 with 441 Male / 142 Female, so a single indicator is
- Severe right skew and heavy outlier counts in V3, V4, V5, V6, V7 (66–84 IQR → Do not apply plain standardization and then trust a linear model; use
- Class imbalance: 416 (71.4%) vs 167 (28.6%) in the Class column → Use stratified train/test and stratified CV folds,
- Ambiguous target polarity — Class is coded 1/2 rather than 0/1 → Explicitly set the positive class (e.g. map 1='liver patient') before
- Very small sample (583 rows, 10 predictors) plus sex imbalance (441 Male / → Use repeated stratified k-fold CV (e.g. 5x5) with reported
- Columns are anonymized (V1–V10), so derived-feature semantics depend on the → Confirm the column-to-biomarker mapping and units with the data owner

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
