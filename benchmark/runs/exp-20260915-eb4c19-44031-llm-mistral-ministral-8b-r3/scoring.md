# Scoring — exp-20260915-eb4c19-44031-llm-mistral-ministral-8b-r3

Dataset: openml-44031 (california)
Status: completed

known_issues: 1
correct_detections: 1  ['CENSOR-44031']
missed_issues: 0  []
unverified_findings: 8

## Findings reported by the tool

- Target 'price' is piled up at its maximum: 965 row(s) (4.7%) sit at exactly
- Latitude,Longitude → Create distance_to_coast feature (e.g., using Geospatial proximity may correlate with price (e.g., coastal properties
- AveRooms,AveBedrms → Derive `rooms_per_bedroom` ratio (AveRooms/AveBedrms) Non-linear relationships between rooms and bedrooms may exist; ratios can
- Population → Apply log transformation or bin into population density Highly skewed distribution (IQR outliers) suggests non-linear impact on
- AveOccup → Flag unrealistic values (e.g., >5 occupants per bedroom) or Extreme outliers (e.g., 1243 occupants) likely indicate data errors or
- MedInc → Bin into income quartiles or interact with `Latitude` (e.g., Income is a strong predictor of housing prices; binning may capture
- Censored/capped target (`price` at 1.79176 with 4.7% of data) → Validate if `price` is truly capped (e.g., survey limits) or treat as
- Outliers in `AveRooms`, `AveBedrms`, `Population`, and `AveOccup` → Apply winsorization (capping extreme values at P99) or use robust
- No explicit target encoding or validation split → Use stratified sampling (if class imbalance exists) or time-based splits
- Geospatial features (`Latitude`, `Longitude`) not leveraged for spatial → Explore kernel density estimation or spatial clustering (e.g., DBSCAN)

## Unmatched findings (unverified, not refuted)

- Latitude,Longitude → Create distance_to_coast feature (e.g., using Geospatial proximity may correlate with price (e.g., coastal properties
- AveRooms,AveBedrms → Derive `rooms_per_bedroom` ratio (AveRooms/AveBedrms) Non-linear relationships between rooms and bedrooms may exist; ratios can
- Population → Apply log transformation or bin into population density Highly skewed distribution (IQR outliers) suggests non-linear impact on
- AveOccup → Flag unrealistic values (e.g., >5 occupants per bedroom) or Extreme outliers (e.g., 1243 occupants) likely indicate data errors or
- MedInc → Bin into income quartiles or interact with `Latitude` (e.g., Income is a strong predictor of housing prices; binning may capture
- Outliers in `AveRooms`, `AveBedrms`, `Population`, and `AveOccup` → Apply winsorization (capping extreme values at P99) or use robust
- No explicit target encoding or validation split → Use stratified sampling (if class imbalance exists) or time-based splits
- Geospatial features (`Latitude`, `Longitude`) not leveraged for spatial → Explore kernel density estimation or spatial clustering (e.g., DBSCAN)

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
