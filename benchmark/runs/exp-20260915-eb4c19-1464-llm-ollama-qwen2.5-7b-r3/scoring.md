# Scoring — exp-20260915-eb4c19-1464-llm-ollama-qwen2.5-7b-r3

Dataset: openml-1464 (blood-transfusion-service-center)
Configuration: llm:ollama-qwen2.5-7b
Status: completed
Scoring version: 2.0; ground-truth registry 1.1

Scope scored: LLM output only, from the first of ✨ Recommended models, 🔧 Feature engineering, ⚠ Pitfalls onward.

known_issues: 1
correct_detections: 0  []
missed_issues: 1  ['DUP-1464']
unverified_findings: 6  (= 2 unmatched detection claims + 4 suggestions)

A detection claim asserts a defect and is checked against the registry;
it comes from `⚠ Warnings` or `⚠ Pitfalls`. A suggestion comes from
`🔧 Feature engineering` or `✨ Recommended models` and has no ground
truth to match against, so it is counted apart rather than pooled with
claims that could have matched and did not.

## Findings in scope

- V1 → log(V1 + 1) Log transformation can stabilize variance for skewed data.
- V2 → z_score(V2) Z-score normalization helps in comparing variables with different scales.
- V3 → log(V3 + 1) V3 has a wide range, log transformation can linearize the relationship and
- V4 → z_score(V4) Similar to V2, z-score normalization helps in handling scale differences
- Class imbalance (76.2% vs 23.8%) → Use AUC/F1, class_weight='balanced', or focal loss
- Possible duplicate rows (28.7%) → Deduplicate data before splitting into train and test sets

## Unmatched detection claims (unverified, not refuted)

- Class imbalance (76.2% vs 23.8%) → Use AUC/F1, class_weight='balanced', or focal loss
- Possible duplicate rows (28.7%) → Deduplicate data before splitting into train and test sets

## Suggestions (no ground truth to match against)

- V1 → log(V1 + 1) Log transformation can stabilize variance for skewed data.
- V2 → z_score(V2) Z-score normalization helps in comparing variables with different scales.
- V3 → log(V3 + 1) V3 has a wide range, log transformation can linearize the relationship and
- V4 → z_score(V4) Similar to V2, z-score normalization helps in handling scale differences

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['DUP-1464']

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
