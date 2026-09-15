# Scoring — exp-20260915-eb4c19-1480-llm-mistral-ministral-8b-r1

Dataset: openml-1480 (ilpd)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1480']
missed_issues: 0  []
unverified_findings: 8

## Findings reported by the tool

- 13 exact duplicate row(s) (2.2% of the data). A random split puts copies of
- V1 → Standardize or normalize (e.g., using `StandardScaler` or Mean (44.7) and std (16.2) suggest a wide range; scaling improves
- V3, V4, V5, V6, V7 → Apply log transformation or Winsorization to mitigate High IQR/z-score outliers (e.g., V3: 84 IQR outliers) may skew model
- V8, V9, V10 → Bin into discrete ranges (e.g., low/medium/high) or use Narrow ranges (e.g., V8: min=2.7, max=9.6) may benefit from non-linear
- V2 (gender) → One-hot encode explicitly (avoid ordinal assumptions) and Categorical with 2 levels; interaction terms may capture gender-specific
- All numeric columns → Create interaction terms (e.g., `V1*V3`, `V5/V6`) to High variance (e.g., V5 std=243) suggests potential non-linear
- Duplicate rows (13 exact duplicates, 2.2% of data) → Remove duplicates before train-test split or use a group key (e.g.,
- Class imbalance (Class=2: 28.6%) → Use class_weight='balanced' in scikit-learn or focal loss; evaluate with
- Outliers in V3–V7 (e.g., V3: 84 IQR outliers) → Validate if outliers are genuine (e.g., measurement errors) or apply
- Small dataset (n=583) → Use cross-validation (e.g., 5-fold) and avoid overfitting; prefer

## Unmatched findings (unverified, not refuted)

- V1 → Standardize or normalize (e.g., using `StandardScaler` or Mean (44.7) and std (16.2) suggest a wide range; scaling improves
- V3, V4, V5, V6, V7 → Apply log transformation or Winsorization to mitigate High IQR/z-score outliers (e.g., V3: 84 IQR outliers) may skew model
- V8, V9, V10 → Bin into discrete ranges (e.g., low/medium/high) or use Narrow ranges (e.g., V8: min=2.7, max=9.6) may benefit from non-linear
- V2 (gender) → One-hot encode explicitly (avoid ordinal assumptions) and Categorical with 2 levels; interaction terms may capture gender-specific
- All numeric columns → Create interaction terms (e.g., `V1*V3`, `V5/V6`) to High variance (e.g., V5 std=243) suggests potential non-linear
- Class imbalance (Class=2: 28.6%) → Use class_weight='balanced' in scikit-learn or focal loss; evaluate with
- Outliers in V3–V7 (e.g., V3: 84 IQR outliers) → Validate if outliers are genuine (e.g., measurement errors) or apply
- Small dataset (n=583) → Use cross-validation (e.g., 5-fold) and avoid overfitting; prefer

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
