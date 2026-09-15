# Scoring — exp-20260915-eb4c19-44031-llm-ollama-qwen2.5-7b-r1

Dataset: openml-44031 (california)
Configuration: llm:ollama-qwen2.5-7b
Status: completed
Scoring version: 2.0; ground-truth registry 1.1

Scope scored: LLM output only, from the first of ✨ Recommended models, 🔧 Feature engineering, ⚠ Pitfalls onward.

known_issues: 1
correct_detections: 1  ['CENSOR-44031']
missed_issues: 0  []
unverified_findings: 2  (= 0 unmatched detection claims + 2 suggestions)

A detection claim asserts a defect and is checked against the registry;
it comes from `⚠ Warnings` or `⚠ Pitfalls`. A suggestion comes from
`🔧 Feature engineering` or `✨ Recommended models` and has no ground
truth to match against, so it is counted apart rather than pooled with
claims that could have matched and did not.

## Findings in scope

- AveRooms → derive room_density per household, (AveRooms / AveOccup) Interactions between number of rooms and occupancy might be important
- Latitude, Longitude → calculate distance to major cities or landmarks as Geographical positions can significantly influence property values
- Target 'price' is capped at its maximum: 965 rows (4.7%) sit exactly at the → Consider capping or flooring target values during modeling;

## Unmatched detection claims (unverified, not refuted)

- (none)

## Suggestions (no ground truth to match against)

- AveRooms → derive room_density per household, (AveRooms / AveOccup) Interactions between number of rooms and occupancy might be important
- Latitude, Longitude → calculate distance to major cities or landmarks as Geographical positions can significantly influence property values

## Inherited from the deterministic layer — context, not counted

These were printed before the advisor was called, so they are not
this configuration's output. They are shown so nothing is hidden.

Detections inherited: ['CENSOR-44031']

- Target 'price' is piled up at its maximum: 965 row(s) (4.7%) sit at exactly

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
