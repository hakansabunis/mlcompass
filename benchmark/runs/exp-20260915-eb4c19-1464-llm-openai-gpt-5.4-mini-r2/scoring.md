# Scoring — exp-20260915-eb4c19-1464-llm-openai-gpt-5.4-mini-r2

Dataset: openml-1464 (blood-transfusion-service-center)
Status: completed

known_issues: 1
correct_detections: 1  ['DUP-1464']
missed_issues: 0  []
unverified_findings: 9

## Findings reported by the tool

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies
- V1 → Consider log1p or quantile-based scaling, and optionally bin into V1 is numeric with a long right tail and a few outliers; recency-style
- V2 → Create interaction features with V1 and V3, and consider ratios such V2 is a count-like numeric variable that may work best in combination with
- V3 → Apply log1p scaling and try bucketed versions (e.g., low/medium/high V3 has a very wide range and clear outliers, so compressed representations
- V4 → Standardize if using linear models; also test interactions with V4 is numeric with a broader continuous range and may carry a different
- V1,V2,V3,V4 → Add pairwise interaction terms and non-linear transforms The four predictors are all numeric and likely encode related
- Exact duplicate rows (215 duplicates, 28.7% of the data). → Verify whether duplicates are genuine repeated records; if not, remove
- Potentially inflated evaluation from random train/test splits because → Deduplicate first or perform grouped/blocked splitting; otherwise
- Class imbalance in Class (about 76% vs 24%). → Use stratified splitting, monitor balanced metrics such as
- Small dataset size with only 748 rows. → Prefer simpler models first, use cross-validation, and keep feature
- Outliers in V1, V2, and V3. → Inspect whether extreme values are valid domain cases; if they are, use

## Unmatched findings (unverified, not refuted)

- V1 → Consider log1p or quantile-based scaling, and optionally bin into V1 is numeric with a long right tail and a few outliers; recency-style
- V2 → Create interaction features with V1 and V3, and consider ratios such V2 is a count-like numeric variable that may work best in combination with
- V3 → Apply log1p scaling and try bucketed versions (e.g., low/medium/high V3 has a very wide range and clear outliers, so compressed representations
- V4 → Standardize if using linear models; also test interactions with V4 is numeric with a broader continuous range and may carry a different
- V1,V2,V3,V4 → Add pairwise interaction terms and non-linear transforms The four predictors are all numeric and likely encode related
- Potentially inflated evaluation from random train/test splits because → Deduplicate first or perform grouped/blocked splitting; otherwise
- Class imbalance in Class (about 76% vs 24%). → Use stratified splitting, monitor balanced metrics such as
- Small dataset size with only 748 rows. → Prefer simpler models first, use cross-validation, and keep feature
- Outliers in V1, V2, and V3. → Inspect whether extreme values are valid domain cases; if they are, use

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
