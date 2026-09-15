# Scoring — exp-20260915-f8ba2f-1464-llm-deepseek-flash-r1

Dataset: openml-1464 (blood-transfusion-service-center)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1464']
missed_issues: 0  []
unverified_findings: 11

## Findings reported by the tool

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies
- V3 → log1p transform (and optionally a per-donation average = V3 / V2) V3 (total blood donated) has mean 1378.7 and std 1459.8 with max 12500 vs
- V1 → keep raw, and add a reciprocal or exponential decay term (e.g. V1 (months since last donation) is the classic recency driver of repeat
- V2 → derive donation rate = V2 / V4 and average interval = V4 / V2 V2 (number of donations, mean 5.5) and V4 (months since first donation,
- V4 → cross with V1 and V2 rather than transforming alone V4 has zero outliers (min 2, max 98, q50 28) and is fairly well behaved on
- V1 → flag whether V1 exceeds the donor's own typical interval (V1 > V4/V2) A donor is 'overdue' when recency V1 exceeds their personal average gap
- Class → map to a 0/1 integer and treat label '1' (570 rows) as the Class is stored as int64 with values {1, 2}; verifying the positive class
- 215 exact duplicate rows (28.7% of the data) flagged in warnings → De-duplicate before splitting, or if copies are genuine repeated
- Class imbalance (76.2% vs 23.8%) → Use stratified train/test splits and stratified k-fold, report
- Very small dataset (748 rows, 4 features) makes validation scores → Use repeated stratified k-fold (e.g. 5x5), keep models
- V2 and V3 contain 45 IQR outliers each (V3 max 12500 vs q50 1000) → Do not naively drop them — these are plausibly heavy donors, not errors.
- Feature names are anonymized (V1-V4) → Confirm the semantics (this is the blood-transfusion layout: recency,
- Any scaling/imputation must be fit inside the CV fold → Wrap StandardScaler (and any log/clip transform) in a Pipeline so it is

## Unmatched findings (unverified, not refuted)

- V3 → log1p transform (and optionally a per-donation average = V3 / V2) V3 (total blood donated) has mean 1378.7 and std 1459.8 with max 12500 vs
- V1 → keep raw, and add a reciprocal or exponential decay term (e.g. V1 (months since last donation) is the classic recency driver of repeat
- V2 → derive donation rate = V2 / V4 and average interval = V4 / V2 V2 (number of donations, mean 5.5) and V4 (months since first donation,
- V4 → cross with V1 and V2 rather than transforming alone V4 has zero outliers (min 2, max 98, q50 28) and is fairly well behaved on
- V1 → flag whether V1 exceeds the donor's own typical interval (V1 > V4/V2) A donor is 'overdue' when recency V1 exceeds their personal average gap
- Class → map to a 0/1 integer and treat label '1' (570 rows) as the Class is stored as int64 with values {1, 2}; verifying the positive class
- Class imbalance (76.2% vs 23.8%) → Use stratified train/test splits and stratified k-fold, report
- Very small dataset (748 rows, 4 features) makes validation scores → Use repeated stratified k-fold (e.g. 5x5), keep models
- V2 and V3 contain 45 IQR outliers each (V3 max 12500 vs q50 1000) → Do not naively drop them — these are plausibly heavy donors, not errors.
- Feature names are anonymized (V1-V4) → Confirm the semantics (this is the blood-transfusion layout: recency,
- Any scaling/imputation must be fit inside the CV fold → Wrap StandardScaler (and any log/clip transform) in a Pipeline so it is

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
