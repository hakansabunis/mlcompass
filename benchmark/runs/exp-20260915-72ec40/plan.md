# Experiment plan — exp-20260915-72ec40

Frozen at 2026-09-15T07:01:50Z (UTC), before execution, per protocol.md section 1.

- Protocol version: 1.0
- Ground-truth registry: ground_truth.json v1.0, frozen 2026-09-15
- mlcompass commit: 787b57cbc2974ab296edf08f7b1e91d2c20f15e0
- Tree dirty at planning time: True
- Python 3.13.3 on Windows-11-10.0.26200-SP0
- Timeout per run: 600s
- Repetitions per cell: 1; seeds [0]
- Reviewers: 1

## Configurations

- `deterministic`: advise with no LLM layer — the rule-based detectors alone.

## Cells

| dataset | configuration | repeats | seeds |
| --- | --- | --- | --- |
| openml-1464 | deterministic | 1 | 0 |
| openml-1480 | deterministic | 1 | 0 |
| openml-44031 | deterministic | 1 | 0 |

## Scoring

An issue counts as detected when its match_pattern (case-insensitive regex) occurs in the scoring scope, AND every value in required_values appears in the same output. The value requirement is what makes a detection specific rather than topical: naming the defect without its magnitude does not identify the defect.

Scope: Combined stdout of the frozen command.

false_positives and hallucinations are not scored automatically; see
ground_truth.json `out_of_scope` for why.

## Execution order

Sequential, dataset-major then configuration, on the recorded environment.
Every repetition starts in a fresh temp workspace with no `.mlcompass/`.
