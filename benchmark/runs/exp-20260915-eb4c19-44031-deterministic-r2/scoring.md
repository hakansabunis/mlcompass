# Scoring — exp-20260915-eb4c19-44031-deterministic-r2

Dataset: openml-44031 (california)
Configuration: deterministic
Status: completed
Scoring version: 2.0; ground-truth registry 1.1

Scope scored: full output.

known_issues: 1
correct_detections: 1  ['CENSOR-44031']
missed_issues: 0  []
unverified_findings: 0  (= 0 unmatched detection claims + 0 suggestions)

A detection claim asserts a defect and is checked against the registry;
it comes from `⚠ Warnings` or `⚠ Pitfalls`. A suggestion comes from
`🔧 Feature engineering` or `✨ Recommended models` and has no ground
truth to match against, so it is counted apart rather than pooled with
claims that could have matched and did not.

## Findings in scope

- Target 'price' is piled up at its maximum: 965 row(s) (4.7%) sit at exactly

## Unmatched detection claims (unverified, not refuted)

- (none)

## Suggestions (no ground truth to match against)

- (none)

false_positives and hallucinations are blank: refuting a reported
issue needs a reviewer, and protocol.md section 4 puts an unrefutable
claim in unverified_findings rather than in false_positives.
