# Scoring — exp-20260915-eb4c19-1480-llm-ollama-qwen2.5-7b-r3

Dataset: openml-1480 (ilpd)
Configuration: llm:ollama-qwen2.5-7b
Status: completed
Scoring version: 2.0; ground-truth registry 1.1

Scope scored: LLM output only, from the first of ✨ Recommended models, 🔧 Feature engineering, ⚠ Pitfalls onward.

known_issues: 1
correct_detections: 0  []
missed_issues: 1  ['DUP-1480']
unverified_findings: 6  (= 2 unmatched detection claims + 4 suggestions)

A detection claim asserts a defect and is checked against the registry;
it comes from `⚠ Warnings` or `⚠ Pitfalls`. A suggestion comes from
`🔧 Feature engineering` or `✨ Recommended models` and has no ground
truth to match against, so it is counted apart rather than pooled with
claims that could have matched and did not.

## Findings in scope

- V3 → log1p(V3) to linearize positive right-skewed distribution, handling Outliers exist (84 iqr_count).
- V4 → log1p(V4) for similar reasons as V3 Outliers exist (81 iqr_count)
- V5 → scale(V5) to maintain feature range, which can be important for High std and outliers present(69 iqr_count).
- V7 → log1p(V7) for similar reasons as V3 and V4 Outliers exist (66 iqr_count).
- Class Imbalance (71% vs. 29%) → Use AUC or F1 score, class_weight='balanced', or consider oversampling
- Duplicate rows (2.2% of the data) → Remove exact duplicates before splitting into train and test sets to

## Unmatched detection claims (unverified, not refuted)

- Class Imbalance (71% vs. 29%) → Use AUC or F1 score, class_weight='balanced', or consider oversampling
- Duplicate rows (2.2% of the data) → Remove exact duplicates before splitting into train and test sets to

## Suggestions (no ground truth to match against)

- V3 → log1p(V3) to linearize positive right-skewed distribution, handling Outliers exist (84 iqr_count).
- V4 → log1p(V4) for similar reasons as V3 Outliers exist (81 iqr_count)
- V5 → scale(V5) to maintain feature range, which can be important for High std and outliers present(69 iqr_count).
- V7 → log1p(V7) for similar reasons as V3 and V4 Outliers exist (66 iqr_count).

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['DUP-1480']

- 13 exact duplicate row(s) (2.2% of the data). A random split puts copies of

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
