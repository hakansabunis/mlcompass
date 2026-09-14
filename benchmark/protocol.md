# Evaluation protocol

Protocol version: 1.0 (template; no experiments conducted).

## 1. Freeze the experiment plan

Before execution, save a dated plan under `runs/<experiment_id>/plan.md`. Record:

- Dataset/case IDs, ground-truth registry snapshot, and issue matching rules.
- Full MLCompass commit, dependency versions, Python version, OS, and hardware.
- Configurations: exact commands, input files, prompts, context, provider,
  requested and resolved model identifiers, and all available generation settings.
- A common seed schedule, repetition count, timeout, retry policy, tool/call
  limits, cache policy, and execution order for every comparable configuration.
- The output scope being scored (for example, the complete final report),
  reviewers, and any numeric tolerances used for claim verification.

Use the same task instructions and resource limits across providers/models.
Record unsupported settings and unavoidable differences; do not silently replace
models. Choose repetition counts before seeing outcomes, and apply the same
count to each comparable dataset/configuration pair. A seed does not guarantee
deterministic responses from hosted models.

## 2. Prepare identical inputs

Follow `datasets.md` to retrieve and checksum a fixed dataset version. Save exact
preparation commands, transformations, seeds, and split indices. Include any
scripts, predictions, or logs needed by the selected MLCompass command.

Use identical prepared artifacts across configurations within each case. Apply
the same preparation rules across datasets, documenting task-specific differences
in advance. Do not provide ground truth or expected findings to MLCompass.
Evaluate materially different tasks/commands separately.

## 3. Execute and preserve evidence

1. Assign a unique `run_id` and create `runs/<run_id>/`.
2. Start from a fresh workspace and identical initial `.mlcompass/` state for
   every repetition. Prevent memory or cached outputs from previous runs from
   affecting later runs, according to the frozen cache policy.
3. Execute the frozen command without manual hints or outcome-driven changes.
   Run sequentially on the recorded environment to avoid resource contention.
4. Measure elapsed wall-clock time using a monotonic clock, from command start
   through completion or termination, including tool calls and retries. Exclude
   dataset download, installation, and preparation from this measurement.
5. Preserve command/configuration, UTC start time, inputs or retrievable hashes,
   stdout/stderr, final output, exit code, errors, retry history, and available
   provider usage metadata. Remove credentials from saved records.
6. Record `status` as `completed`, `failed`, or `timeout`. Never discard a failed
   attempt or silently rerun it; any planned rerun receives a new run ID and a
   reference to the original attempt.

## 4. Score against frozen ground truth

Review the prespecified output scope, splitting compound issue reports into
distinct issue claims and deduplicating repeated claims about the same issue
and affected scope. Record each claim, output location, evidence, label, and
matching ground-truth ID in `runs/<run_id>/scoring.md`.

| Result column | Definition |
| --- | --- |
| `known_issues` | Number of verified, in-scope ground-truth issues. |
| `correct_detections` | Ground-truth issues detected with the required specificity; each issue earns credit at most once. |
| `missed_issues` | Ground-truth issues not detected: `known_issues - correct_detections`. |
| `false_positives` | Distinct reported issues shown by evidence to be absent or incorrect. |
| `unverified_findings` | Distinct reported issues that cannot be confirmed or refuted; absence from the issue list alone is insufficient for a false positive. |
| `hallucinations` | Distinct factual claims contradicted by or unsupported by the supplied input/evidence, such as invented columns, measurements, or executed actions. |

Clearly qualified suggestions to investigate are not asserted detections or
hallucinations. Count hallucinated factual claims separately from issue labels:
a false-positive issue can also contain a hallucination, so these counts must
not be summed as disjoint errors. Apply claim splitting and deduplication rules
consistently to both structured output and free text within the scoring scope.

Use two independent reviewers where feasible, blind to provider/model identity,
and retain their initial labels and adjudicated disagreements. If only one
reviewer is available, state this limitation. Keep unresolved claims unverified.
If new evidence requires ground-truth revision, version the registry and rescore
all affected runs consistently; preserve the original annotations.

For failed or timed-out runs, leave detection and hallucination counts blank;
record known issue count, observed runtime, available usage, and failure details.
Do not treat missing outputs as successful clean reports.

## 5. Account for usage and cost

Sum input and output tokens across all LLM calls, including retries and tool-loop
calls. `total_tokens` is their sum when both are available. Preserve provider
definitions and raw usage, including cached or reasoning-token categories; note
differences that limit comparisons. Leave unavailable values blank rather than
guessing token usage from text length.

Estimate LLM cost in USD using the provider's applicable pricing at execution
time. Preserve the pricing URL, retrieval date, per-category rates, token counts,
and calculation in the run record. For simple input/output billing:

`estimated_llm_cost_usd = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000`

Here rates are USD per million tokens. Where billing distinguishes cache reads,
cache writes, reasoning, or other charges, use the documented categories without
double counting. Leave cost blank if rates or usage are unavailable and explain
why. This is an LLM cost estimate, excluding local compute and dataset costs.

## 6. Record and report

Append one row per run to `results.csv`, linking its artifact directory. Preserve
the frozen plan, environment/configuration record, outputs, usage/pricing record,
and scoring notes so another evaluator can repeat the procedure.

Report counts per dataset/configuration and all planned repetitions. Recall is
`correct_detections / known_issues`. Verified-issue precision is
`correct_detections / (correct_detections + false_positives)`; always report
unverified findings alongside it, since they are excluded. Undefined ratios
remain blank. Do not calculate a false-positive rate without a defined negative
issue universe. Report failures/timeouts separately and state the number of
completed runs used for any quality summary. Runtime, token, and cost summaries
must state their included statuses and missing-data counts. Any later aggregation
or uncertainty method must be specified before examining comparative outcomes.
