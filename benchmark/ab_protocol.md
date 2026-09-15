# A/B protocol — does mlcompass change how an LLM trains a model?

Protocol version: A/B 1.0. No experiments conducted under it yet.

This is a **second** protocol, not a revision of `protocol.md`. That one
asks whether mlcompass detects known defects in a dataset. This one asks a
different question with a different outcome measure, and mixing them would
make both harder to read.

> **The question.** Given the same dataset and the same model, does an LLM
> asked to train a model produce better work when mlcompass's deterministic
> findings are in front of it than when they are not?

## 1. Why this needs its own outcome measure

`protocol.md` scores against a frozen registry of known issues, and its
ground truth is a list. Here the deliverable is a training script and the
model it produces, so the ground truth is **what the script does when it
runs**. That is measurable without a reviewer, which is the reason to
prefer it: every number below is computed by executing the artefact, not
by judging it.

Three outcomes, in priority order:

| Outcome | How it is measured | Why it is first/second/third |
| --- | --- | --- |
| `holdout_score` | The script's model scored on a held-out split the script never sees, prepared by the harness | The only measure that cannot be argued with. A model is better or it is not. |
| `process_defects` | Count of defects in the emitted script, from a frozen checklist (below) | A script can score well and still be wrong — leakage inflates the very number we are reading. |
| `runs_at_all` | Exit status of the emitted script | A sophisticated script that crashes is worth less than a plain one that runs. |

## 2. The two arms

Both arms get the same dataset, the same task description, the same model,
the same generation settings, and the same execution environment. They
differ in exactly one thing.

**`control`** — the LLM is given the raw CSV path, the target column name,
and the task: write a training script.

**`treatment`** — identical, plus the output of `mlcompass advise` and
`mlcompass audit` on the same inputs, pasted into the prompt verbatim.

Nothing else may differ. In particular the treatment prompt must not add
advice of its own: if mlcompass says nothing about a dataset, the treatment
arm sees an empty findings block, not a hint.

## 3. Held-out preparation

The harness, not the LLM, owns the split. Before either arm runs it splits
the pinned CSV once with a recorded seed, writes `train.csv`, and keeps
`holdout.csv` to itself. Both arms are given `train.csv` only, and neither
is told a holdout exists — an arm that knows it is being scored on a
specific file can fit it.

The score is computed by the harness by loading the emitted model (or
re-running the script's predict path) against `holdout.csv`. Metric per
task type, fixed in advance: ROC AUC for binary classification, macro F1
for multiclass, R² for regression.

## 4. Process-defect checklist, frozen

Each is a property of the emitted script, decided mechanically. A defect
counts once per script however many times it occurs.

| Defect | Decided by |
| --- | --- |
| `no_seed` | No seeding call reaches any framework the script imports |
| `leak_fit_before_split` | A scaler, encoder or imputer is fitted before the split, or on the full frame |
| `leak_duplicate_rows` | Duplicates present in the input are not removed before splitting |
| `wrong_metric_for_imbalance` | Accuracy is the reported metric on a target whose minority class is under 20% |
| `no_validation` | No held-out or cross-validated evaluation of any kind inside the script |
| `target_in_features` | The target column remains in the feature matrix |

The first five are what `mlcompass audit` and `advise` already check, which
makes the comparison fair in the direction that matters: the treatment arm
is being handed findings about exactly these, so the measure is whether
being told helps, not whether the checklist favours the tool. Scoring the
checklist is done by a frozen script, never by the model that wrote the
code and never by a model at all.

## 5. Repetitions and what may be concluded

LLM output varies run to run, so a single pair proves nothing. Repetitions
per (dataset x arm x model) are fixed before execution, as in `protocol.md`
section 1, and every repetition is kept including failures.

Report per model and per dataset. **Do not pool across models**: this is
the same constraint `analysis_plan_2026-09.md` section 3.1 imposes for the
contract battery, and for the same reason — a difference that appears on a
7B local model and not on a frontier one is a finding, not noise to be
averaged away. Pooling would hide the result most worth having, which is
whether mlcompass helps a weak model more than a strong one.

## 6. The losing condition, written before the data

State it now so it cannot be renegotiated later:

> If the treatment arm's held-out score is not better than the control's,
> and its process-defect count is not lower, then on these datasets and
> these models mlcompass did not improve the work. That is the result, it
> gets reported in those words, and the protocol is not revised to find a
> measure under which it looks better.

A plausible and publishable outcome is that the scores are indistinguishable
while the defect count falls. That would say mlcompass improves the
*process* without improving the *metric*, which is worth stating precisely
rather than dressing up as either a win or a failure.

## 7. Evidence

Same discipline as `protocol.md` section 3: unique run id, fresh workspace,
the exact prompt, the emitted script, its stdout and exit code, the split
seed, the holdout score, and the defect-checklist output, all preserved
under `runs/<run_id>/`. The emitted script is kept verbatim — it is the
artefact under study and cannot be reconstructed from a score.
