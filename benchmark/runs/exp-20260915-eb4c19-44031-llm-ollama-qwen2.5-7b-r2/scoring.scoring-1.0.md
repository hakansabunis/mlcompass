# Scoring — exp-20260915-eb4c19-44031-llm-ollama-qwen2.5-7b-r2

Dataset: openml-44031 (california)
Status: completed

known_issues: 1
correct_detections: 1  ['CENSOR-44031']
missed_issues: 0  []
unverified_findings: 4

## Findings reported by the tool

- Target 'price' is piled up at its maximum: 965 row(s) (4.7%) sit at exactly
- AveRooms → derive log(AveRooms) Reduces skewness in the distribution
- Population → log(Population) or sqrt(Population) Often helps with large skewed values
- ['Latitude', 'Longitude'] → derive distance_from_centroid; use PCA for Reduces sparsity and multicollinearity
- 'price' values are capped at exactly 1.79176 → Adjust the model to account for censored data or remove capped data
- Missing interaction features between geographic columns → Engage in spatial analysis techniques; include derived fields like

## Unmatched findings (unverified, not refuted)

- AveRooms → derive log(AveRooms) Reduces skewness in the distribution
- Population → log(Population) or sqrt(Population) Often helps with large skewed values
- ['Latitude', 'Longitude'] → derive distance_from_centroid; use PCA for Reduces sparsity and multicollinearity
- Missing interaction features between geographic columns → Engage in spatial analysis techniques; include derived fields like

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
