# Scoring — exp-20260915-eb4c19-44031-llm-mistral-ministral-8b-r1

Dataset: openml-44031 (california)
Status: completed

known_issues: 1
correct_detections: 1  ['CENSOR-44031']
missed_issues: 0  []
unverified_findings: 5

## Findings reported by the tool

- Target 'price' is piled up at its maximum: 965 row(s) (4.7%) sit at exactly
- Latitude,Longitude → Create `distance_to_coast` or `distance_to_center` Geographic proximity may correlate with price; raw coordinates alone lack
- AveRooms,AveBedrms → Derive `rooms_per_bedroom` ratio (AveRooms/AveBedrms) Outliers in `AveRooms` (e.g., 141.9) suggest extreme values; ratios often
- Population → Bin into `population_quartiles` or use log-scaling (log1p) to High skew (IQR=938) and outliers (e.g., 35682) may distort linear models;
- price → Treat as censored data: split into `price_below_cap` (0/1) and Warning indicates potential censoring; standard regression assumes
- Censored target (`price` at 1.79176) → Use Tobit regression or split into binary/censored models; avoid
- Outliers in `AveRooms`, `AveBedrms`, `Population` → Apply Winsorization (e.g., 99th percentile cap) or robust scaling (e.g.,
- No explicit validation split or train-test separation → Reserve 20% for testing early; use stratified sampling if `price`

## Unmatched findings (unverified, not refuted)

- Latitude,Longitude → Create `distance_to_coast` or `distance_to_center` Geographic proximity may correlate with price; raw coordinates alone lack
- AveRooms,AveBedrms → Derive `rooms_per_bedroom` ratio (AveRooms/AveBedrms) Outliers in `AveRooms` (e.g., 141.9) suggest extreme values; ratios often
- Population → Bin into `population_quartiles` or use log-scaling (log1p) to High skew (IQR=938) and outliers (e.g., 35682) may distort linear models;
- Outliers in `AveRooms`, `AveBedrms`, `Population` → Apply Winsorization (e.g., 99th percentile cap) or robust scaling (e.g.,
- No explicit validation split or train-test separation → Reserve 20% for testing early; use stratified sampling if `price`

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
