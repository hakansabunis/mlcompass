# Scoring — exp-20260915-f8ba2f-44031-llm-deepseek-flash-r1

Dataset: openml-44031 (california)
Status: completed

known_issues: 1
correct_detections: 1  ['CENSOR-44031']
missed_issues: 0  []
unverified_findings: 8

## Findings reported by the tool

- Target 'price' is piled up at its maximum: 965 row(s) (4.7%) sit at exactly
- Latitude, Longitude → Add rotated coordinates (e.g. 45° rotation), distance Latitude and Longitude are the only geospatial columns and their raw
- MedInc → Use log1p(MedInc) and add interactions with geo cluster or MedInc is the strongest classic predictor (mean 3.87, max 15.0, 681 IQR
- AveRooms, AveBedrms → Add the ratio AveRooms/AveBedrms and BedroomsPerRoom AveBedrms shows the most IQR outliers (1,424) and AveRooms reaches 141.9,
- Population, AveOccup → Derive Households = Population / AveOccup, and AveOccup has max 1243 in a column whose median is 2.82 — plausible only if
- price → Add a binary is_capped flag (price == 1.79176) and consider 965 rows (4.7%) sit at exactly the maximum 1.79176, so the target is
- Censored target: 4.7% of 'price' rows (965) are pinned at the maximum → Fit a Tobit / censored regression, or add an is_capped indicator and
- Heavy-tailed predictors: AveOccup (max 1243), Population (max 35,682), → Log-transform or winsorize these columns and use scale-robust models
- Spatial autocorrelation and repeated coordinates: rows are aggregated block → Use spatially grouped cross-validation (e.g. group by rounded
- Latitude/Longitude used as plain linear features in a linear model, where → Feed splines, rotations, or geo-cluster one-hots into the linear
- Derived-column leakage: AveRooms and AveBedrms already embed household → Build the ratio and Households features explicitly, then check pairwise

## Unmatched findings (unverified, not refuted)

- Latitude, Longitude → Add rotated coordinates (e.g. 45° rotation), distance Latitude and Longitude are the only geospatial columns and their raw
- MedInc → Use log1p(MedInc) and add interactions with geo cluster or MedInc is the strongest classic predictor (mean 3.87, max 15.0, 681 IQR
- AveRooms, AveBedrms → Add the ratio AveRooms/AveBedrms and BedroomsPerRoom AveBedrms shows the most IQR outliers (1,424) and AveRooms reaches 141.9,
- Population, AveOccup → Derive Households = Population / AveOccup, and AveOccup has max 1243 in a column whose median is 2.82 — plausible only if
- Heavy-tailed predictors: AveOccup (max 1243), Population (max 35,682), → Log-transform or winsorize these columns and use scale-robust models
- Spatial autocorrelation and repeated coordinates: rows are aggregated block → Use spatially grouped cross-validation (e.g. group by rounded
- Latitude/Longitude used as plain linear features in a linear model, where → Feed splines, rotations, or geo-cluster one-hots into the linear
- Derived-column leakage: AveRooms and AveBedrms already embed household → Build the ratio and Households features explicitly, then check pairwise

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
