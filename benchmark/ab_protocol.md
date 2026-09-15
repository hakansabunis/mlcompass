# A/B protocol — does mlcompass change how an LLM trains a model?

Protocol version: A/B 1.1. No scored experiments conducted under it yet;
one unscored pipeline-validation pair was run under 1.0 and is recorded in
§9. The three amendments in §9 were made before any scored run, prompted by
gaps the validation pair exposed.

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

## 2. The three arms

Every arm gets the same dataset, the same task description, the same model,
the same generation settings, and the same execution environment. Each
differs from the one above it in exactly one thing.

**`control`** — the LLM is given the raw CSV path, the target column name,
and the task: write a training script.

**`advise`** — identical, plus the verbatim output of `mlcompass advise` on
the same inputs.

**`advise+audit`** — the `advise` arm, then one revision round: the script it
produced is passed through `mlcompass audit`, the findings are returned to the
same model, and it revises. Scoring uses the revised script.

Nothing else may differ. In particular no arm's prompt may add advice of its
own: if mlcompass says nothing, the arm sees an empty findings block, not a
hint. Each arm audits only its **own** script, so the arms stay independent —
feeding one arm's output to another would make them a single two-step
condition rather than three comparable ones.

Why `audit` needs its own arm rather than sharing `advise`'s: the two tools
read different things at different times. `advise` reads the *data* and can
run before any code exists. `audit` reads a *training script* and cannot run
until one does. Asking for both in the initial prompt is not underspecified,
it is impossible — and the distinction is not cosmetic. `advise` tells the
model something about the dataset, which it is free to ignore, and in the
validation pair it did exactly that: handed a duplicate-row warning, it
produced a script that does not de-duplicate. `audit` tells the model
something about the code it just wrote, which is harder to wave away. If
mlcompass changes the work anywhere, this is the likelier place, and a
two-arm design would have measured the weaker half and reported the result
as the tool's.

## 3. Held-out preparation

The harness, not the LLM, owns the split. Before any arm runs it splits the
pinned CSV once with a recorded seed, writes `train.csv`, and keeps
`holdout.csv` to itself, outside the workspace the script can see. Every arm
is given `train.csv` only and none is told a holdout exists — an arm that
knows it is being scored on a specific file can fit it.

**Geometry, frozen:** holdout fraction **0.25**, and **stratified on the
target for classification tasks**. The fraction is the usual compromise —
smaller and the score is noise, larger and the model is starved and looks
worse than it is. Stratification is not a preference: on 1464 the minority
class is 23.8%, and an unstratified draw can leave the holdout with a single
class present, where ROC AUC is undefined and the run is wasted for a reason
that has nothing to do with the arms.

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

**Generation settings, frozen: temperature 1.0 on every arm and every
provider**, recorded in each run record. What matters is not the number but
that it is the same everywhere and written down — provider defaults differ,
so leaving it unset would mean a difference between lanes could come from a
setting rather than from a model, with no way to tell afterwards which.

Temperature 0 was the alternative and is rejected deliberately. It would make
runs nearly identical and the repetitions redundant, and run-to-run variance
is a measurement this study wants: the detection benchmark has already shown
one lane looking calm on a single draw and erratic across three. Fixing
temperature to 0 would not answer that question; it would stop it being
askable.

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

## 8. Losing condition, restated for three arms

§6 was written for two. With three the statement is the same in shape and
has to be made for each step, so that a null result on one cannot be quietly
carried by the other:

> If `advise` does not beat `control` on held-out score and does not lower
> the defect count, then being told about the data did not improve the work.
> If `advise+audit` does not beat `advise` on either measure, then the
> revision round did not either. Each is reported in those words for the
> step it concerns. Neither result licenses a claim about the other, and the
> protocol is not revised to find a measure under which either looks better.

The outcome this design exists to separate: `advise` failing while
`advise+audit` succeeds would say mlcompass helps by checking the code, not
by describing the data. Reporting that as one undifferentiated "mlcompass
helps" would be the more flattering claim and the less true one.

## 9. Amendments

Recorded per `protocol.md` §1 discipline: all three were made **before any
scored run**, prompted by what a single unscored validation pair exposed.
That pair — local `qwen2.5:7b`, openml-1464, one repetition per arm — is
kept as evidence of the pipeline working end to end and is **not** a result:
both arms scored ROC AUC 0.6835 with 1 of 6 defects, at n=1.

**A1 — third arm (`advise+audit`).** Under 1.0 the treatment arm was
specified to carry both `advise` and `audit` output. That is not
underspecified but impossible: `audit` reads a training script, and none
exists when the initial prompt is built. Rather than invent one — a starter
script would be an intervention of our own, and the control arm's output
would make the arms dependent — `audit` moves to its own arm as a revision
round, which is also where a person would use it. Prediction: the defect
count falls from `advise` to `advise+audit` more than from `control` to
`advise`. Rejected if the `advise+audit` defect count is not lower than
`advise`'s.

**A2 — split geometry frozen** at 0.25 and stratified-for-classification.
Under 1.0 §3 fixed that the harness owns the split and records the seed, but
not the fraction or the stratification. Unstratified draws on an imbalanced
target can make the score undefined, which is a wasted run for a reason
unrelated to the arms.

**A3 — temperature frozen at 1.0** across arms and providers. Under 1.0 §2
required "the same generation settings" without naming one, so each provider
supplied its own default and a between-lane difference could have come from
the setting rather than the model.
