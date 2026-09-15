# Scoring — exp-20260915-eb4c19-44031-llm-deepseek-flash-r3

Dataset: openml-44031 (california)
Status: completed

known_issues: 1
correct_detections: 1  ['CENSOR-44031']
missed_issues: 0  []
unverified_findings: 12

## Findings reported by the tool

- Target 'price' is piled up at its maximum: 965 row(s) (4.7%) sit at exactly
- Latitude + Longitude → Derive distance-to-coast and distance to major These two geo columns have no missing values and near-symmetric
- MedInc → Add log(MedInc), percentile rank, and interactions with the geo MedInc is right-skewed (mean 3.87 vs median 3.53, max 15.0, 681 IQR
- AveRooms + AveBedrms → Create AveBedrms/AveRooms ratio, AveRooms minus Both are extreme-tailed (AveRooms max 141.9 vs median 5.23; AveBedrms max
- AveOccup → Use log1p(AveOccup) or a rank transform, and optionally AveOccup has the worst tail in the dataset (max 1243.3 vs median 2.82, std
- Population → Add log(Population) and a derived household count = Population Population is heavily right-skewed (max 35682 vs median 1166) and,
- Population + AveRooms + AveOccup → Cross-column: total_rooms = AveRooms * AveRooms, AveOccup and Population are district-level averages, so explicit
- HouseAge → Bin into decade buckets or add HouseAge x geo interactions HouseAge spans 1-52 with no outliers (IQR/z-score counts both 0) and is
- Target is piled up at its maximum: 965 rows (4.7%) sit at exactly 1.79176, → Treat as censored regression (Tobit) or fit a two-stage model: a
- Heavy-tailed outliers in AveOccup (max 1243.3), AveRooms (max 141.9), → Winsorize at 1st/99th percentile, use log1p/rank transforms, and scale
- Rows are census districts, so AveRooms/AveBedrms/AveOccup/Population are → Interpret coefficients as district-level; consider population-weighted
- Spatial autocorrelation and geographic leakage risk if you build → Use out-of-fold/KFold target encoding for any geo-derived feature, and
- Possible duplicate district records (a known characteristic of this dataset → De-duplicate on the full feature set (or on rounded lat/lon plus
- Small feature count (8 predictors, 20640 rows) makes it easy to over-fit → Use k-fold cross-validation, keep tree depth/regularization tuned, and

## Unmatched findings (unverified, not refuted)

- Latitude + Longitude → Derive distance-to-coast and distance to major These two geo columns have no missing values and near-symmetric
- MedInc → Add log(MedInc), percentile rank, and interactions with the geo MedInc is right-skewed (mean 3.87 vs median 3.53, max 15.0, 681 IQR
- AveRooms + AveBedrms → Create AveBedrms/AveRooms ratio, AveRooms minus Both are extreme-tailed (AveRooms max 141.9 vs median 5.23; AveBedrms max
- AveOccup → Use log1p(AveOccup) or a rank transform, and optionally AveOccup has the worst tail in the dataset (max 1243.3 vs median 2.82, std
- Population → Add log(Population) and a derived household count = Population Population is heavily right-skewed (max 35682 vs median 1166) and,
- Population + AveRooms + AveOccup → Cross-column: total_rooms = AveRooms * AveRooms, AveOccup and Population are district-level averages, so explicit
- HouseAge → Bin into decade buckets or add HouseAge x geo interactions HouseAge spans 1-52 with no outliers (IQR/z-score counts both 0) and is
- Heavy-tailed outliers in AveOccup (max 1243.3), AveRooms (max 141.9), → Winsorize at 1st/99th percentile, use log1p/rank transforms, and scale
- Rows are census districts, so AveRooms/AveBedrms/AveOccup/Population are → Interpret coefficients as district-level; consider population-weighted
- Spatial autocorrelation and geographic leakage risk if you build → Use out-of-fold/KFold target encoding for any geo-derived feature, and
- Possible duplicate district records (a known characteristic of this dataset → De-duplicate on the full feature set (or on rounded lat/lon plus
- Small feature count (8 predictors, 20640 rows) makes it easy to over-fit → Use k-fold cross-validation, keep tree depth/regularization tuned, and

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
