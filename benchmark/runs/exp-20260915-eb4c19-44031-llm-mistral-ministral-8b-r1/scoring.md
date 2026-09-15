# Scoring — exp-20260915-eb4c19-44031-llm-mistral-ministral-8b-r1

Dataset: openml-44031 (california)
Configuration: llm:mistral-ministral-8b
Status: completed
Scoring version: 2.0; ground-truth registry 1.1

Scope scored: LLM output only, from the first of ✨ Recommended models, 🔧 Feature engineering, ⚠ Pitfalls onward.

known_issues: 1
correct_detections: 0  []
missed_issues: 1  ['CENSOR-44031']
unverified_findings: 7  (= 3 unmatched detection claims + 4 suggestions)

A detection claim asserts a defect and is checked against the registry;
it comes from `⚠ Warnings` or `⚠ Pitfalls`. A suggestion comes from
`🔧 Feature engineering` or `✨ Recommended models` and has no ground
truth to match against, so it is counted apart rather than pooled with
claims that could have matched and did not.

## Findings in scope

- Latitude,Longitude → Create `distance_to_coast` or `distance_to_center` Geographic proximity may correlate with price; raw coordinates alone lack
- AveRooms,AveBedrms → Derive `rooms_per_bedroom` ratio (AveRooms/AveBedrms) Outliers in `AveRooms` (e.g., 141.9) suggest extreme values; ratios often
- Population → Bin into `population_quartiles` or use log-scaling (log1p) to High skew (IQR=938) and outliers (e.g., 35682) may distort linear models;
- price → Treat as censored data: split into `price_below_cap` (0/1) and Warning indicates potential censoring; standard regression assumes
- Censored target (`price` at 1.79176) → Use Tobit regression or split into binary/censored models; avoid
- Outliers in `AveRooms`, `AveBedrms`, `Population` → Apply Winsorization (e.g., 99th percentile cap) or robust scaling (e.g.,
- No explicit validation split or train-test separation → Reserve 20% for testing early; use stratified sampling if `price`

## Unmatched detection claims (unverified, not refuted)

- Censored target (`price` at 1.79176) → Use Tobit regression or split into binary/censored models; avoid
- Outliers in `AveRooms`, `AveBedrms`, `Population` → Apply Winsorization (e.g., 99th percentile cap) or robust scaling (e.g.,
- No explicit validation split or train-test separation → Reserve 20% for testing early; use stratified sampling if `price`

## Suggestions (no ground truth to match against)

- Latitude,Longitude → Create `distance_to_coast` or `distance_to_center` Geographic proximity may correlate with price; raw coordinates alone lack
- AveRooms,AveBedrms → Derive `rooms_per_bedroom` ratio (AveRooms/AveBedrms) Outliers in `AveRooms` (e.g., 141.9) suggest extreme values; ratios often
- Population → Bin into `population_quartiles` or use log-scaling (log1p) to High skew (IQR=938) and outliers (e.g., 35682) may distort linear models;
- price → Treat as censored data: split into `price_below_cap` (0/1) and Warning indicates potential censoring; standard regression assumes

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['CENSOR-44031']

- Target 'price' is piled up at its maximum: 965 row(s) (4.7%) sit at exactly

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
