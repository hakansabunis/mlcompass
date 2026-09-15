# Experiment plan — exp-20260915-f8ba2f

Frozen at 2026-09-15T09:36:57Z (UTC), before execution, per protocol.md section 1.

- Protocol version: 1.0
- Ground-truth registry: ground_truth.json v1.0, frozen 2026-09-15
- mlcompass commit: 2075fdeb0862b5f6008fc84c6872e8ce834d69c8
- Tree dirty at planning time: True
- Python 3.13.3 on Windows-11-10.0.26200-SP0
- Timeout per run: 600s
- Repetitions per cell: 1; seeds [0]
- Reviewers: 1

## Configurations

- `deterministic`: advise with no LLM layer — the rule-based detectors alone.
- `llm:deepseek-flash`: advise with the LLM advisor on top of the same deterministic analysis. Model `deepseek-flash` via `openai` at https://api.deepseek.com — the published battery's provider family.
- `llm:mistral-ministral-8b`: advise with the LLM advisor on top of the same deterministic analysis. Model `ministral-8b-latest` via `openai` at https://api.mistral.ai/v1 — fourth provider, small hosted model.
- `llm:ollama-qwen2.5-7b`: advise with the LLM advisor on top of the same deterministic analysis. Model `qwen2.5:7b` via `openai` at http://localhost:11434/v1 — local, free, no credential.
- `llm:openai-gpt-5.4-mini`: advise with the LLM advisor on top of the same deterministic analysis. Model `gpt-5.4-mini` via `openai` at the provider's default endpoint — the pre-registered strict-capable closed lane.

## Cells

| dataset | configuration | repeats | seeds |
| --- | --- | --- | --- |
| openml-1464 | deterministic | 1 | 0 |
| openml-1464 | llm:deepseek-flash | 1 | 0 |
| openml-1464 | llm:mistral-ministral-8b | 1 | 0 |
| openml-1464 | llm:ollama-qwen2.5-7b | 1 | 0 |
| openml-1464 | llm:openai-gpt-5.4-mini | 1 | 0 |
| openml-1480 | deterministic | 1 | 0 |
| openml-1480 | llm:deepseek-flash | 1 | 0 |
| openml-1480 | llm:mistral-ministral-8b | 1 | 0 |
| openml-1480 | llm:ollama-qwen2.5-7b | 1 | 0 |
| openml-1480 | llm:openai-gpt-5.4-mini | 1 | 0 |
| openml-44031 | deterministic | 1 | 0 |
| openml-44031 | llm:deepseek-flash | 1 | 0 |
| openml-44031 | llm:mistral-ministral-8b | 1 | 0 |
| openml-44031 | llm:ollama-qwen2.5-7b | 1 | 0 |
| openml-44031 | llm:openai-gpt-5.4-mini | 1 | 0 |

## Scoring

An issue counts as detected when its match_pattern (case-insensitive regex) occurs in the scoring scope, AND every value in required_values appears in the same output. The value requirement is what makes a detection specific rather than topical: naming the defect without its magnitude does not identify the defect.

Scope: Combined stdout of the frozen command.

false_positives and hallucinations are not scored automatically; see
ground_truth.json `out_of_scope` for why.

## Execution order

Sequential, dataset-major then configuration, on the recorded environment.
Every repetition starts in a fresh temp workspace with no `.mlcompass/`.
