# Scoring — exp-20260915-eb4c19-44031-llm-openai-gpt-5.4-mini-r2

Dataset: openml-44031 (california)
Configuration: llm:openai-gpt-5.4-mini
Status: completed
Scoring version: 2.0; ground-truth registry 1.1

Scope scored: LLM output only, from the first of ✨ Recommended models, 🔧 Feature engineering, ⚠ Pitfalls onward.

known_issues: 1
correct_detections: 0  []
missed_issues: 1  ['CENSOR-44031']
unverified_findings: 11  (= 5 unmatched detection claims + 6 suggestions)

A detection claim asserts a defect and is checked against the registry;
it comes from `⚠ Warnings` or `⚠ Pitfalls`. A suggestion comes from
`🔧 Feature engineering` or `✨ Recommended models` and has no ground
truth to match against, so it is counted apart rather than pooled with
claims that could have matched and did not.

## Findings in scope

- Latitude, Longitude → Create spatial features such as latitude/longitude The target is housing-related and location variables are present;
- AveRooms, AveBedrms → Engineer ratios such as bedrooms_per_room, These two columns are related by construction and the raw levels include
- Population, AveOccup → Apply log1p transforms and/or cap extreme values; Both columns show heavy skew/outliers, especially AveOccup and Population,
- MedInc → Add nonlinear transforms such as log(MedInc) and quantile/bin MedInc is usually a strong predictor in California housing-style data and
- HouseAge → Try spline, bucketed, or quadratic features for age bands. HouseAge has a bounded range (1 to 52) and may relate to price nonlinearly
- Latitude, Longitude → Add interaction features with MedInc and HouseAge, Price effects often vary by region; the same income level can map
- Capped/censored target at the upper bound → Inspect how price was recorded and consider censored-regression
- Severe outliers in several predictors → Use robust preprocessing such as log transforms, winsorization, or
- Potential leakage from using raw target-derived or post-outcome variables → Verify that none of the predictors are computed using information
- Need to evaluate with scale-appropriate regression metrics and residual → Use MAE/RMSE and inspect residuals by target level and by geographic
- Possible multicollinearity among housing composition variables → Check correlations among MedInc, AveRooms, AveBedrms, Population, and

## Unmatched detection claims (unverified, not refuted)

- Capped/censored target at the upper bound → Inspect how price was recorded and consider censored-regression
- Severe outliers in several predictors → Use robust preprocessing such as log transforms, winsorization, or
- Potential leakage from using raw target-derived or post-outcome variables → Verify that none of the predictors are computed using information
- Need to evaluate with scale-appropriate regression metrics and residual → Use MAE/RMSE and inspect residuals by target level and by geographic
- Possible multicollinearity among housing composition variables → Check correlations among MedInc, AveRooms, AveBedrms, Population, and

## Suggestions (no ground truth to match against)

- Latitude, Longitude → Create spatial features such as latitude/longitude The target is housing-related and location variables are present;
- AveRooms, AveBedrms → Engineer ratios such as bedrooms_per_room, These two columns are related by construction and the raw levels include
- Population, AveOccup → Apply log1p transforms and/or cap extreme values; Both columns show heavy skew/outliers, especially AveOccup and Population,
- MedInc → Add nonlinear transforms such as log(MedInc) and quantile/bin MedInc is usually a strong predictor in California housing-style data and
- HouseAge → Try spline, bucketed, or quadratic features for age bands. HouseAge has a bounded range (1 to 52) and may relate to price nonlinearly
- Latitude, Longitude → Add interaction features with MedInc and HouseAge, Price effects often vary by region; the same income level can map

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['CENSOR-44031']

- Target 'price' is piled up at its maximum: 965 row(s) (4.7%) sit at exactly

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
