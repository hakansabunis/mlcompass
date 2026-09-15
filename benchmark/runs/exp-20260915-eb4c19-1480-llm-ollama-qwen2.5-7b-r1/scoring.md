# Scoring — exp-20260915-eb4c19-1480-llm-ollama-qwen2.5-7b-r1

Dataset: openml-1480 (ilpd)
Configuration: llm:ollama-qwen2.5-7b
Status: completed
Scoring version: 2.0; ground-truth registry 1.1

Scope scored: LLM output only, from the first of ✨ Recommended models, 🔧 Feature engineering, ⚠ Pitfalls onward.

known_issues: 1
correct_detections: 0  []
missed_issues: 1  ['DUP-1480']
unverified_findings: 5  (= 2 unmatched detection claims + 3 suggestions)

A detection claim asserts a defect and is checked against the registry;
it comes from `⚠ Warnings` or `⚠ Pitfalls`. A suggestion comes from
`🔧 Feature engineering` or `✨ Recommended models` and has no ground
truth to match against, so it is counted apart rather than pooled with
claims that could have matched and did not.

## Findings in scope

- V3 → log(V3) or Standardize High variance, logarithmic transformation might help
- V4 → log(V4) or Standardize High variance, logarithmic transformation might help
- V5 → log(V5) or Winsorization High variance and outliers, winzorizing the high values could be useful
- Class imbalance (71.4% of one class) → Use AUC/F1 as metrics; apply SMOTE to generate synthetic data or use
- Duplicate rows → Deduplicate the dataset before splitting into training and test sets

## Unmatched detection claims (unverified, not refuted)

- Class imbalance (71.4% of one class) → Use AUC/F1 as metrics; apply SMOTE to generate synthetic data or use
- Duplicate rows → Deduplicate the dataset before splitting into training and test sets

## Suggestions (no ground truth to match against)

- V3 → log(V3) or Standardize High variance, logarithmic transformation might help
- V4 → log(V4) or Standardize High variance, logarithmic transformation might help
- V5 → log(V5) or Winsorization High variance and outliers, winzorizing the high values could be useful

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['DUP-1480']

- 13 exact duplicate row(s) (2.2% of the data). A random split puts copies of

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
