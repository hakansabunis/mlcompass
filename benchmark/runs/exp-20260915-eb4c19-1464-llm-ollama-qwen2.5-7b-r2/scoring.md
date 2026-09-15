# Scoring — exp-20260915-eb4c19-1464-llm-ollama-qwen2.5-7b-r2

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

- V1 → normalize or standardize ... wide range, might benefit from scaling for consistent learning
- V2 → log transformation ... skewed distribution with outliers, could help stabilize variance and
- V3 → scale or normalize ... very wide range compared to others, benefits from scaling for
- V4 → quantization or binning ... few outliers already removed, potential value in discretizing into
- Class imbalance (76.2% vs 23.8%) → use AUC/F1, class_weight='balanced', or focal loss to handle imbalance
- Highly exact duplicate rows (28.7%) → de-duplicate the data before splitting into training and testing sets

## Unmatched detection claims (unverified, not refuted)

- Class imbalance (76.2% vs 23.8%) → use AUC/F1, class_weight='balanced', or focal loss to handle imbalance
- Highly exact duplicate rows (28.7%) → de-duplicate the data before splitting into training and testing sets

## Suggestions (no ground truth to match against)

- V1 → normalize or standardize ... wide range, might benefit from scaling for consistent learning
- V2 → log transformation ... skewed distribution with outliers, could help stabilize variance and
- V3 → scale or normalize ... very wide range compared to others, benefits from scaling for
- V4 → quantization or binning ... few outliers already removed, potential value in discretizing into

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['DUP-1464']

- 215 exact duplicate row(s) (28.7% of the data). A random split puts copies

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
