# Scoring — exp-20260915-f8ba2f-44031-llm-mistral-ministral-8b-r1

Dataset: openml-44031 (california)
Status: completed

known_issues: 1
correct_detections: 1  ['CENSOR-44031']
missed_issues: 0  []
unverified_findings: 5

## Findings reported by the tool

- Target 'price' is piled up at its maximum: 965 row(s) (4.7%) sit at exactly
- MedInc → Apply log transformation (e.g., `log1p(MedInc)`) or bin into High skew (mean > std) and outliers (681 IQR outliers) suggest
- AveRooms → Create ratio features like `AveRoomsPerBedroom = AveRooms / Both columns show extreme outliers (133 Z-score outliers for `AveRooms`,
- ['Latitude', 'Longitude'] → Derive `distance_to_coast` (if coastal data is Geospatial patterns (e.g., proximity to amenities) may correlate with
- price → Treat as censored data: model only values below the cap (1.79176) 4.7% of observations are piled at the max value, indicating potential
- Censored target (`price` capped at 1.79176) → Validate data collection process (e.g., survey limits) and consider
- Outliers in `AveRooms`, `AveBedrms`, and `Population` → Use robust scaling (e.g., `RobustScaler`) or outlier-resistant models
- No explicit train/validation split or feature leakage risk → Stratify splits by geographic regions (lat/lon) to avoid spatial

## Unmatched findings (unverified, not refuted)

- MedInc → Apply log transformation (e.g., `log1p(MedInc)`) or bin into High skew (mean > std) and outliers (681 IQR outliers) suggest
- AveRooms → Create ratio features like `AveRoomsPerBedroom = AveRooms / Both columns show extreme outliers (133 Z-score outliers for `AveRooms`,
- ['Latitude', 'Longitude'] → Derive `distance_to_coast` (if coastal data is Geospatial patterns (e.g., proximity to amenities) may correlate with
- Outliers in `AveRooms`, `AveBedrms`, and `Population` → Use robust scaling (e.g., `RobustScaler`) or outlier-resistant models
- No explicit train/validation split or feature leakage risk → Stratify splits by geographic regions (lat/lon) to avoid spatial

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
