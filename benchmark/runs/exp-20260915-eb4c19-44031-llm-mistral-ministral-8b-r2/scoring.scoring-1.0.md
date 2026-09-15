# Scoring — exp-20260915-eb4c19-44031-llm-mistral-ministral-8b-r2

Dataset: openml-44031 (california)
Status: completed

known_issues: 1
correct_detections: 1  ['CENSOR-44031']
missed_issues: 0  []
unverified_findings: 6

## Findings reported by the tool

- Target 'price' is piled up at its maximum: 965 row(s) (4.7%) sit at exactly
- Latitude,Longitude → Create `distance_to_coast` (e.g., via Haversine Geospatial proximity to coastlines or regional clusters (e.g., urban vs.
- AveRooms,AveBedrms → Derive `rooms_per_bedroom` (ratio) and `log(AveRooms)` High outliers in `AveRooms` (e.g., 141.9) suggest potential data errors;
- Population → Bin into `population_quartiles` or use `log1p(Population)` to Population ranges from 3 to 35,682; log-transforms or binning can better
- price → Treat as censored data: split into `price_below_cap` (≤1.79) and Warning indicates 4.7% of observations are capped; ignoring this may bias
- Censored target (`price` capped at 1.79176) → Use Tobit regression or split into uncapped/capped subproblems; avoid
- Outliers in `AveRooms`/`AveBedrms` (e.g., 141.9 rooms) → Investigate outliers (e.g., data entry errors) or cap/winorize extreme
- No explicit train/test split or validation strategy → Reserve 20% for validation; use cross-validation to account for small
- No target encoding or feature interactions tested → Test interactions like `Latitude*Population` or embed lat/lon as

## Unmatched findings (unverified, not refuted)

- Latitude,Longitude → Create `distance_to_coast` (e.g., via Haversine Geospatial proximity to coastlines or regional clusters (e.g., urban vs.
- AveRooms,AveBedrms → Derive `rooms_per_bedroom` (ratio) and `log(AveRooms)` High outliers in `AveRooms` (e.g., 141.9) suggest potential data errors;
- Population → Bin into `population_quartiles` or use `log1p(Population)` to Population ranges from 3 to 35,682; log-transforms or binning can better
- Outliers in `AveRooms`/`AveBedrms` (e.g., 141.9 rooms) → Investigate outliers (e.g., data entry errors) or cap/winorize extreme
- No explicit train/test split or validation strategy → Reserve 20% for validation; use cross-validation to account for small
- No target encoding or feature interactions tested → Test interactions like `Latitude*Population` or embed lat/lon as

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
