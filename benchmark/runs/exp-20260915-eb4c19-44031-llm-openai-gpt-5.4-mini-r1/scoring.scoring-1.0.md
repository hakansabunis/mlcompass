# Scoring — exp-20260915-eb4c19-44031-llm-openai-gpt-5.4-mini-r1

Dataset: openml-44031 (california)
Status: completed

known_issues: 1
correct_detections: 1  ['CENSOR-44031']
missed_issues: 0  []
unverified_findings: 8

## Findings reported by the tool

- Target 'price' is piled up at its maximum: 965 row(s) (4.7%) sit at exactly
- Latitude, Longitude → Create spatial features such as latitude/longitude The target 'price' is likely strongly location-dependent, and raw
- MedInc → Try log transform or winsorization and add interaction terms with MedInc has a wide range and outliers, and income usually interacts
- AveRooms, AveBedrms, Population, AveOccup → Engineer ratio features such as These columns are already averages/count-like measures, so derived ratios
- HouseAge → Add non-linear transforms or binning (e.g., age bands, squared HouseAge spans 1 to 52 with no missingness; the relationship to 'price' is
- All numeric predictors → Standardize features for linear models and Several predictors show strong skew/outliers (notably 'Population',
- Capped/censored target at the maximum value → Investigate why 965 rows have 'price' exactly at 1.79176; if it is a
- Heavy outliers and extreme skew in several predictors → Use robust preprocessing, log transforms, winsorization, or tree-based
- Potential leakage-free but non-iid spatial structure → Use spatially aware validation splits or at least check performance by
- Average-based features may hide underlying denominators → Verify how 'AveRooms', 'AveBedrms', and 'AveOccup' were constructed and

## Unmatched findings (unverified, not refuted)

- Latitude, Longitude → Create spatial features such as latitude/longitude The target 'price' is likely strongly location-dependent, and raw
- MedInc → Try log transform or winsorization and add interaction terms with MedInc has a wide range and outliers, and income usually interacts
- AveRooms, AveBedrms, Population, AveOccup → Engineer ratio features such as These columns are already averages/count-like measures, so derived ratios
- HouseAge → Add non-linear transforms or binning (e.g., age bands, squared HouseAge spans 1 to 52 with no missingness; the relationship to 'price' is
- All numeric predictors → Standardize features for linear models and Several predictors show strong skew/outliers (notably 'Population',
- Heavy outliers and extreme skew in several predictors → Use robust preprocessing, log transforms, winsorization, or tree-based
- Potential leakage-free but non-iid spatial structure → Use spatially aware validation splits or at least check performance by
- Average-based features may hide underlying denominators → Verify how 'AveRooms', 'AveBedrms', and 'AveOccup' were constructed and

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
