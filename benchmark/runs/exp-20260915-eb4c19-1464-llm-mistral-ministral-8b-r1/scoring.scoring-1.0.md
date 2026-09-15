# Scoring — exp-20260915-eb4c19-1464-llm-mistral-ministral-8b-r1

Dataset: openml-1464 (blood-transfusion-service-center)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1464']
missed_issues: 0  []
unverified_findings: 7

## Findings reported by the tool

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies
- V1 → Apply log-transform or winsorize outliers (IQR-based) to reduce skew High zero_ratio (0.67%) and outliers (7 IQR/Z-score) suggest skewed
- V2 → Bin into quartiles (e.g., 1–2, 3–4, 5–7, 8+) to reduce noise from 45 45 IQR outliers and sparse high-end values (max=50) imply a long tail;
- V3 → Normalize via log(V3 + 1) or scale to [0,1] using min-max (min=250, Mean (1378.68) vs. median (1000) and high std (1459.83) indicate
- V4 → No transformation needed (no outliers), but consider interaction terms V4’s distribution is clean, but V1’s outliers may interact with V4’s
- Class → Resample (oversample class=2 or undersample class=1) or use Class imbalance (76% vs. 24%) risks biased models; explicit handling
- Duplicate rows (215 exact copies, 28.7% of data) → De-duplicate via `drop_duplicates()` before train-test split, or split
- No explicit feature engineering for temporal patterns (e.g., V1–V4 as → If V1–V4 represent sequential measurements (e.g., lab values), add lag
- Lack of domain knowledge to interpret V1–V4 labels → Consult subject-matter experts to validate if V1–V4 are truly

## Unmatched findings (unverified, not refuted)

- V1 → Apply log-transform or winsorize outliers (IQR-based) to reduce skew High zero_ratio (0.67%) and outliers (7 IQR/Z-score) suggest skewed
- V2 → Bin into quartiles (e.g., 1–2, 3–4, 5–7, 8+) to reduce noise from 45 45 IQR outliers and sparse high-end values (max=50) imply a long tail;
- V3 → Normalize via log(V3 + 1) or scale to [0,1] using min-max (min=250, Mean (1378.68) vs. median (1000) and high std (1459.83) indicate
- V4 → No transformation needed (no outliers), but consider interaction terms V4’s distribution is clean, but V1’s outliers may interact with V4’s
- Class → Resample (oversample class=2 or undersample class=1) or use Class imbalance (76% vs. 24%) risks biased models; explicit handling
- No explicit feature engineering for temporal patterns (e.g., V1–V4 as → If V1–V4 represent sequential measurements (e.g., lab values), add lag
- Lack of domain knowledge to interpret V1–V4 labels → Consult subject-matter experts to validate if V1–V4 are truly

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
