# Scoring — exp-20260915-eb4c19-44031-llm-deepseek-flash-r1

Dataset: openml-44031 (california)
Configuration: llm:deepseek-flash
Status: completed
Scoring version: 2.0; ground-truth registry 1.1

Scope scored: LLM output only, from the first of ✨ Recommended models, 🔧 Feature engineering, ⚠ Pitfalls onward.

known_issues: 1
correct_detections: 1  ['CENSOR-44031']
missed_issues: 0  []
unverified_findings: 13  (= 5 unmatched detection claims + 8 suggestions)

A detection claim asserts a defect and is checked against the registry;
it comes from `⚠ Warnings` or `⚠ Pitfalls`. A suggestion comes from
`🔧 Feature engineering` or `✨ Recommended models` and has no ground
truth to match against, so it is counted apart rather than pooled with
claims that could have matched and did not.

## Findings in scope

- Latitude → Build geo features jointly with Longitude: KMeans clusters Latitude and Longitude are the only spatial columns (no outliers, ranges
- Longitude → Add rotated coordinates (e.g. rotate lat/lon by ~30-45 degrees) The California coastline runs diagonally; rotated axes align with the
- AveRooms → Winsorize or log-transform, and derive rooms_per_person = AveRooms has IQR outliers (511 rows) and a max of 141.9 versus a median of
- AveBedrms → Winsorize at the 99th percentile (or log1p) before use in the AveBedrms shows the most IQR outliers of any column (1424 rows) with max
- AveOccup → Log-transform or cap, and replace/augment with households = AveOccup is extremely right-skewed (mean 3.07, max 1243.3, std 10.4, 711
- Population → Use log1p(Population) and Population / AveOccup; consider Population spans 3 to 35,682 with 1196 IQR outliers, and no area column
- MedInc → Add log(MedInc), MedInc × AveRooms, and MedInc relative to the KNN MedInc has 681 IQR outliers and a max of 15.0 against a median of 3.53,
- HouseAge → Bin into coarse age bands and interact with the geo cluster; add HouseAge is the cleanest column (0 outliers, range 1-52, median 29) so it
- Target 'price' is capped: 965 rows (4.7%) sit at exactly 1.79176, matching → Confirm how the value was recorded before treating it as ordinary
- Spatial autocorrelation: rows are geographic (Latitude/Longitude, 20,640 → Use spatially blocked or grouped cross-validation (cluster on
- Small-denominator artifacts create the extreme values in AveRooms (max → Do not drop as 'errors'. Winsorize or log-transform for the linear
- Scale mismatch between feature types: Population is in the thousands while → Fit a StandardScaler/RobustScaler inside the CV pipeline for the linear
- No missing values are reported in any of the 9 columns, which is unusual → Confirm the provenance of the file before assuming missingness is a
- Potential duplicate or near-duplicate rows (block groups sharing identical → Check for exact duplicates on the coordinate pair plus Population before

## Unmatched detection claims (unverified, not refuted)

- Spatial autocorrelation: rows are geographic (Latitude/Longitude, 20,640 → Use spatially blocked or grouped cross-validation (cluster on
- Small-denominator artifacts create the extreme values in AveRooms (max → Do not drop as 'errors'. Winsorize or log-transform for the linear
- Scale mismatch between feature types: Population is in the thousands while → Fit a StandardScaler/RobustScaler inside the CV pipeline for the linear
- No missing values are reported in any of the 9 columns, which is unusual → Confirm the provenance of the file before assuming missingness is a
- Potential duplicate or near-duplicate rows (block groups sharing identical → Check for exact duplicates on the coordinate pair plus Population before

## Suggestions (no ground truth to match against)

- Latitude → Build geo features jointly with Longitude: KMeans clusters Latitude and Longitude are the only spatial columns (no outliers, ranges
- Longitude → Add rotated coordinates (e.g. rotate lat/lon by ~30-45 degrees) The California coastline runs diagonally; rotated axes align with the
- AveRooms → Winsorize or log-transform, and derive rooms_per_person = AveRooms has IQR outliers (511 rows) and a max of 141.9 versus a median of
- AveBedrms → Winsorize at the 99th percentile (or log1p) before use in the AveBedrms shows the most IQR outliers of any column (1424 rows) with max
- AveOccup → Log-transform or cap, and replace/augment with households = AveOccup is extremely right-skewed (mean 3.07, max 1243.3, std 10.4, 711
- Population → Use log1p(Population) and Population / AveOccup; consider Population spans 3 to 35,682 with 1196 IQR outliers, and no area column
- MedInc → Add log(MedInc), MedInc × AveRooms, and MedInc relative to the KNN MedInc has 681 IQR outliers and a max of 15.0 against a median of 3.53,
- HouseAge → Bin into coarse age bands and interact with the geo cluster; add HouseAge is the cleanest column (0 outliers, range 1-52, median 29) so it

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['CENSOR-44031']

- Target 'price' is piled up at its maximum: 965 row(s) (4.7%) sit at exactly

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
