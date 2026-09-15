# Scoring — exp-20260915-eb4c19-44031-llm-deepseek-flash-r2

Dataset: openml-44031 (california)
Configuration: llm:deepseek-flash
Status: completed
Scoring version: 2.0; ground-truth registry 1.1

Scope scored: LLM output only, from the first of ✨ Recommended models, 🔧 Feature engineering, ⚠ Pitfalls onward.

known_issues: 1
correct_detections: 1  ['CENSOR-44031']
missed_issues: 0  []
unverified_findings: 12  (= 4 unmatched detection claims + 8 suggestions)

A detection claim asserts a defect and is checked against the registry;
it comes from `⚠ Warnings` or `⚠ Pitfalls`. A suggestion comes from
`🔧 Feature engineering` or `✨ Recommended models` and has no ground
truth to match against, so it is counted apart rather than pooled with
claims that could have matched and did not.

## Findings in scope

- Latitude → Add engineered spatial features: rotated coordinates (e.g. 'Latitude' and 'Longitude' have no outliers and tight ranges (32.54-41.95,
- 124.35 to -114.31), so location is clearly a clean, high-signal predictor; raw
- Longitude → Pair with 'Latitude' in interaction terms and in a KNN-distance Coastal vs inland price gradients in California are diagonal, not
- MedInc → log1p(MedInc), plus interactions MedInc*Latitude and 'MedInc' is right-skewed with q50 3.53 but max 15.0001 and 681 IQR
- AveRooms → Create ratios: AveRooms/AveOccup (rooms per person), 'AveRooms' has 511 IQR outliers and a max of 141.9 against a q50 of 5.23,
- AveOccup → log1p transform plus winsorization/clipping at a high 'AveOccup' has q75 3.28 but a max of 1243, which almost certainly reflects
- Population → log1p(Population) and derived Population/AveOccup household 'Population' has 1,196 IQR outliers and a max of 35,682 versus q50 1,166,
- HouseAge → Keep as-is; optionally add a binning (deciles) or a flag for 'HouseAge' shows zero IQR and zero z-score outliers and a max of 52 - the
- price → Create a censoring indicator for price == 1.79176 and consider 965 rows (4.7%) sit exactly at the target maximum 1.79176, so the target
- Target 'price' is piled up at its maximum 1.79176 (965 rows, 4.7%), → Fit a censored (Tobit-style) or quantile/robust-loss model, or drop/flag
- Extreme outliers in 'AveRooms' (max 141.9), 'AveBedrms' (1,424 IQR → Log-transform and/or winsorize these columns, add anomaly flags, and
- 'MedInc' (max 15.0001) and 'HouseAge' (max 52) both stop at suspiciously → Treat the top codes as censored: add top-code indicator features rather
- Strong spatial autocorrelation: neighboring census blocks have nearly → Use spatially blocked or grouped cross-validation (e.g. group by
- No missingness is reported, which may mean missing values were silently → Confirm the provenance of the CSV, check for duplicated rows and for
- Target range is narrow (0.14-1.79, std 0.36) and bounded, so plain OLS can → Clip predictions to the observed target range for reporting, and prefer

## Unmatched detection claims (unverified, not refuted)

- Extreme outliers in 'AveRooms' (max 141.9), 'AveBedrms' (1,424 IQR → Log-transform and/or winsorize these columns, add anomaly flags, and
- Strong spatial autocorrelation: neighboring census blocks have nearly → Use spatially blocked or grouped cross-validation (e.g. group by
- No missingness is reported, which may mean missing values were silently → Confirm the provenance of the CSV, check for duplicated rows and for
- Target range is narrow (0.14-1.79, std 0.36) and bounded, so plain OLS can → Clip predictions to the observed target range for reporting, and prefer

## Suggestions (no ground truth to match against)

- Latitude → Add engineered spatial features: rotated coordinates (e.g. 'Latitude' and 'Longitude' have no outliers and tight ranges (32.54-41.95,
- 124.35 to -114.31), so location is clearly a clean, high-signal predictor; raw
- Longitude → Pair with 'Latitude' in interaction terms and in a KNN-distance Coastal vs inland price gradients in California are diagonal, not
- MedInc → log1p(MedInc), plus interactions MedInc*Latitude and 'MedInc' is right-skewed with q50 3.53 but max 15.0001 and 681 IQR
- AveRooms → Create ratios: AveRooms/AveOccup (rooms per person), 'AveRooms' has 511 IQR outliers and a max of 141.9 against a q50 of 5.23,
- AveOccup → log1p transform plus winsorization/clipping at a high 'AveOccup' has q75 3.28 but a max of 1243, which almost certainly reflects
- Population → log1p(Population) and derived Population/AveOccup household 'Population' has 1,196 IQR outliers and a max of 35,682 versus q50 1,166,
- HouseAge → Keep as-is; optionally add a binning (deciles) or a flag for 'HouseAge' shows zero IQR and zero z-score outliers and a max of 52 - the

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['CENSOR-44031']

- Target 'price' is piled up at its maximum: 965 row(s) (4.7%) sit at exactly

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
