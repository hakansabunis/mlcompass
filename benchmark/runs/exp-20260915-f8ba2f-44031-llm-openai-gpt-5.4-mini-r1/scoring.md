# Scoring — exp-20260915-f8ba2f-44031-llm-openai-gpt-5.4-mini-r1

Dataset: openml-44031 (california)
Status: completed

known_issues: 1
correct_detections: 1  ['CENSOR-44031']
missed_issues: 0  []
unverified_findings: 10

## Findings reported by the tool

- Target 'price' is piled up at its maximum: 965 row(s) (4.7%) sit at exactly
- Latitude, Longitude → Add spatial features such as geohash/grid cell, The target is a California housing-related regression problem, and
- MedInc → Try log-transforming or winsorizing extreme values; optionally add ‘MedInc’ has a wide range and visible outliers, which can make linear fits
- AveRooms, AveBedrms → Create derived ratios like rooms_per_bedroom = These columns are structurally related and can capture housing composition
- Population, AveOccup → Add population density-style or crowding features, Both columns show strong skew/outliers, and occupancy-related effects are
- HouseAge → Add nonlinear basis features such as age bins or spline terms. HouseAge spans 1 to 52 and may affect price in a thresholded or saturating
- All numeric predictors → Standardize features for linear models and The feature scales differ substantially across columns, especially
- Capped/censored target values at the upper bound of 'price' → Treat the maximum-pileup as a censoring/measurement issue to
- Heavy right-skew and outliers in several predictors → Use robust preprocessing such as log transforms, winsorization, or
- Potentially misleading evaluation if using random splits only → Use cross-validation and, if possible, spatially aware validation by
- Ratio features can explode on small denominators → When engineering ratios from ‘AveRooms’ and ‘AveBedrms’, add small
- No missing values means quality issues may be hidden rather than absent → Still run sanity checks for duplicates, implausible values, and unit

## Unmatched findings (unverified, not refuted)

- Latitude, Longitude → Add spatial features such as geohash/grid cell, The target is a California housing-related regression problem, and
- MedInc → Try log-transforming or winsorizing extreme values; optionally add ‘MedInc’ has a wide range and visible outliers, which can make linear fits
- AveRooms, AveBedrms → Create derived ratios like rooms_per_bedroom = These columns are structurally related and can capture housing composition
- Population, AveOccup → Add population density-style or crowding features, Both columns show strong skew/outliers, and occupancy-related effects are
- HouseAge → Add nonlinear basis features such as age bins or spline terms. HouseAge spans 1 to 52 and may affect price in a thresholded or saturating
- All numeric predictors → Standardize features for linear models and The feature scales differ substantially across columns, especially
- Heavy right-skew and outliers in several predictors → Use robust preprocessing such as log transforms, winsorization, or
- Potentially misleading evaluation if using random splits only → Use cross-validation and, if possible, spatially aware validation by
- Ratio features can explode on small denominators → When engineering ratios from ‘AveRooms’ and ‘AveBedrms’, add small
- No missing values means quality issues may be hidden rather than absent → Still run sanity checks for duplicates, implausible values, and unit

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
