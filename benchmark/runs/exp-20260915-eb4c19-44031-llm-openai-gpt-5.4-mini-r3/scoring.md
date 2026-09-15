# Scoring — exp-20260915-eb4c19-44031-llm-openai-gpt-5.4-mini-r3

Dataset: openml-44031 (california)
Status: completed

known_issues: 1
correct_detections: 1  ['CENSOR-44031']
missed_issues: 0  []
unverified_findings: 9

## Findings reported by the tool

- Target 'price' is piled up at its maximum: 965 row(s) (4.7%) sit at exactly
- MedInc → consider log-transforming or winsorizing extreme values; It has a wide range and outliers (IQR/z-score counts) in 'MedInc', so a
- AveRooms, AveBedrms, AveOccup, Population → create derived ratios such as These columns show strong skew/outliers, and ratio features often capture
- Latitude + Longitude → engineer spatial features: grid/geohash clusters, The target is California housing 'price', so location is likely a major
- HouseAge → try nonlinear encodings such as splines or age bins 'HouseAge' is bounded (1 to 52) and may relate to price nonlinearly rather
- All numeric predictors → standardize for linear models; for tree models, The numeric scales differ substantially across 'Population', 'Longitude',
- Target censoring/capping at the maximum value → Investigate why 'price' has 965 rows exactly at 1.79176; if it is a cap,
- Heavy-tailed predictors and extreme outliers → Inspect and possibly cap or transform 'AveRooms', 'AveBedrms',
- Potential leakage from inappropriate preprocessing → Fit all transforms, scaling, and outlier thresholds only on training
- Spatial autocorrelation in California housing data → Use spatially aware validation if possible, or at least inspect
- Multicollinearity among density-related predictors → Check correlations among 'AveRooms', 'AveBedrms', 'Population', and

## Unmatched findings (unverified, not refuted)

- MedInc → consider log-transforming or winsorizing extreme values; It has a wide range and outliers (IQR/z-score counts) in 'MedInc', so a
- AveRooms, AveBedrms, AveOccup, Population → create derived ratios such as These columns show strong skew/outliers, and ratio features often capture
- Latitude + Longitude → engineer spatial features: grid/geohash clusters, The target is California housing 'price', so location is likely a major
- HouseAge → try nonlinear encodings such as splines or age bins 'HouseAge' is bounded (1 to 52) and may relate to price nonlinearly rather
- All numeric predictors → standardize for linear models; for tree models, The numeric scales differ substantially across 'Population', 'Longitude',
- Heavy-tailed predictors and extreme outliers → Inspect and possibly cap or transform 'AveRooms', 'AveBedrms',
- Potential leakage from inappropriate preprocessing → Fit all transforms, scaling, and outlier thresholds only on training
- Spatial autocorrelation in California housing data → Use spatially aware validation if possible, or at least inspect
- Multicollinearity among density-related predictors → Check correlations among 'AveRooms', 'AveBedrms', 'Population', and

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
