# A/B protocol — does mlcompass change how an LLM trains a model?

Protocol version: A/B 1.3. No scored experiments conducted under it yet.
Fourteen unscored pipeline-validation runs were made under 1.0 and 1.1, all
of them superseded and all of them kept; they are what the four amendments
in §9 were written from. Every amendment was made **before any scored run**.
§10 records what is known to be wrong with the design and is not being
fixed.

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
the list of packages available in the execution environment, and the task:
write a training script.

The package list is not a courtesy. Without it the model guesses at the
environment, and a script that fails on `import imblearn` has failed for a
reason that says nothing about the quality of the code — it is the harness
scoring its own `pip list`. Worse, the guess is not arm-neutral: mlcompass's
advise output recommends addressing class imbalance, which points a model
towards resampling libraries, so an unstated environment would make the
treatment arms fail more often for a reason unrelated to the intervention.
The same list goes to every arm, and it names what is actually importable
rather than what we would like to be.

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

**Geometry, frozen:** holdout fraction **0.25**, **stratified on the target
for classification tasks**, and **split on duplicate groups** — every set of
identical rows lands entirely on one side. See §9 A4: without the last of
these the scoring rewarded the defect it was built to penalise. The fraction is the usual compromise —
smaller and the score is noise, larger and the model is starved and looks
worse than it is. Stratification is not a preference: on 1464 the minority
class is 23.8%, and an unstratified draw can leave the holdout with a single
class present, where ROC AUC is undefined and the run is wasted for a reason
that has nothing to do with the arms.

Splitting on duplicate groups is not a refinement; it is what makes the
held-out score mean anything on data that contains duplicates. A plain
stratified split of OpenML 1464 puts **71 of 187 holdout rows (38%)
verbatim into train**, so a script that does the right thing and
de-duplicates is scored on a holdout it has been denied the answers to,
while one that leaves the duplicates in is scored on rows it memorised.
Grouping first leaves the holdout sharing no row with train, and leaves
**125 duplicates still inside train**, so `leak_duplicate_rows` remains a
defect a script can commit and the score no longer depends on committing it.

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

Recorded per `protocol.md` §1 discipline: all four were made **before any
scored run**, prompted by what the unscored validation runs exposed. Those
runs — local `qwen2.5:7b`, openml-1464, one repetition per arm — were kept
as evidence of the pipeline working end to end and are **not** results. A1
to A3 came out of the first pair under 1.0, where both arms scored ROC AUC
0.6835 with 1 of 6 defects at n=1; A4 came out of the twelve three-arm runs
under 1.1, which is the set that made the inversion visible. Every one of
them is superseded by A4 and marked as such where it sits.

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

**A4 — split on duplicate groups.** Under 1.1 the harness split the pinned
CSV directly. On OpenML 1464 that placed 71 of 187 holdout rows verbatim in
train, and the consequence was not subtle: across the **twelve** validation
runs recorded under 1.1 the relationship between the checklist and the score
was perfectly inverted and perfectly separated — **every one of the eight
scripts that left the duplicates in scored at least 0.6532, every one of the
four that removed them scored 0.5552**, and no run fell between. The
benchmark was paying scripts to leak, on the one measure §1 calls the one
that cannot be argued with.

Two corrections to the numbers this amendment was first drafted with, made
on re-reading `ab_results.csv` and recorded rather than quietly applied.
There are twelve 1.1 runs, not seven. And the 1-defect runs did not all
score 0.6835: seven did and one scored 0.6532, so the relationship is a
clean separation of two groups rather than a pair of constants. Both
corrections leave the conclusion where it was and make it slightly stronger
— the separation is total, with the worst leaking run still above the best
clean one.

This is the failure mlcompass exists to detect, built into the scoring of
the study meant to evaluate it, and mlcompass's own warning text names the
remedy: de-duplicate before splitting, *or split on a group key*. The second
is chosen because the first would delete the defect along with the leak —
grouping keeps 125 duplicates inside train, so a script can still fail to
remove them, while the holdout shares no row with train and the score is
honest for every arm.

Every validation run recorded before this amendment is superseded. They are
kept, since discarding evidence of a design error is how the error survives.

## 10. Stated limitations

Things known to be wrong with this design, recorded rather than fixed. The
distinction from §9 is that an amendment changes the protocol; a limitation
is a fact about what the protocol can and cannot measure, and writing it down
is the alternative to quietly working around it.

**L1 — on this dataset family the `advise+audit` intervention is an empty
block, so A1's prediction may not be testable here.** `mlcompass audit`
reported "No issues detected by the static checks" on **5 of 5** first-turn
scripts across the 1.1 and 1.2 validation runs, all of them pandas +
scikit-learn on OpenML 1464. The revision round still happened — §2 requires
it, and an audit finding nothing is a result — but what went back to the
model was a framed empty block. An arm whose intervention is empty cannot be distinguished from
the arm above it, and A1 predicted the defect count would fall further from
`advise` to `advise+audit` than from `control` to `advise`.

The mechanism is not that the scripts were clean. `audit`'s `seed` rule
returns early unless the script imports one of a fixed set of stochastic
frameworks — torch, tensorflow, keras, numpy, random, jax, lightning —
and scikit-learn is not among them, so an unseeded pandas + sklearn script
is not flagged. The same script with `import numpy as np` added and nothing
else changed is flagged `error / seed`. Of the remaining rules, `val_split`
does apply to sklearn but only fires when the script performs no split at all,
and `optimizer`, `loss_stability`, `dataloader`, `grad_clipping`, `eval_mode`
and `batch_size` are deep-learning shaped. A tabular sklearn script that calls
`train_test_split` therefore has, in practice, no rule left that can fire.

`audit` is **not** being changed to close this. The checklist in §4 is frozen
and five of its six items are what `audit` and `advise` already check, which
is what makes the comparison fair; widening `audit`'s rules after seeing that
the treatment arm produced no findings would be tuning the instrument to the
experiment, and any subsequent difference between the arms would be
uninterpretable. The honest options are to report the null for this family,
or to add a dataset family where `audit` has rules that apply — a
torch-shaped task — and report that separately. Until one of those happens,
`advise+audit` results on tabular sklearn data are reported with this
limitation attached, and a defect count that does not fall between `advise`
and `advise+audit` is not evidence that revision does not help. It is
evidence that nothing was said.

**A5 — state the execution environment in the prompt.** Under 1.2 the
prompt named the CSV, the target and the columns, but not what was
installed. A validation run's `advise` arm died on `import imblearn`, which
is a reasonable library to reach for and simply absent from this
environment: `runs_at_all`, one of §1's three outcomes, was partly
measuring package availability. The confound is also directional — advise
output recommends handling class imbalance, which points towards exactly
the resampling libraries most likely to be missing — so leaving it unstated
would penalise the treatment arms for taking the intervention's advice.
Every arm now receives the same list of importable packages, generated from
the environment the script will actually run in rather than written by
hand.

**A6 — a saved bundle is not a scoreable artefact, and that is a scorer
limitation rather than a result.** Unresolved; see §6. Recorded here so the
numbering below is not mistaken for a gap.

**A7 — the defect checker reported `target_in_features` on scripts that
exclude the target.** Found by Yusuf Ünlü in review of the 108-run battery,
against the preserved evidence rather than against a claim. The rule scores
the *absence* of every enumerated way of taking the target out, and the
enumeration did not include a comprehension:

```python
target_col = "Class"
feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols]
```

`runs/ab-20260915-3b0221-1464-control-deepseek-flash-r1/emitted.py:11` does
exactly this and was scored `target_in_features: yes`. **Fifteen of the 108
runs carried the false positive.**

The shape of the rule is the defect, not any single missing pattern. An
absence-scored checklist has an open-ended list of forms, so every form
nobody thought of becomes a false positive on correct code — which inflates
the defect count in the direction that flatters the treatment arms, since
correct scripts are what the treatment is supposed to produce. Comprehension
and set-difference forms were added, and `tests/test_ab_target_exclusion.py`
now pins every accepted form together with five scripts that genuinely leave
the target in, because widening an exclusion rule is how a detector becomes
a rubber stamp.

Corrected against the same bytes via `benchmark/rescore_ab.py` — no model
call, no re-execution, dataset facts read back from each run's `run.json`.
Original rows and every original `scoring.md` are kept; the corrected rows
carry experiment id `ab-20260915-3b0221-rescored` and each run gained a
`scoring_rescored.md` beside its original.

| lane | control | advise | advise+audit |
| --- | ---: | ---: | ---: |
| deepseek-flash | 14 → **13** | 15 → **15** | 7 → **5** |
| mistral-ministral-8b | 8 → **8** | 2 → **2** | 0 → **0** |
| ollama-qwen2.5-7b | 6 → **6** | 4 → **4** | 1 → **1** |
| openai-gpt-5.4-mini | 12 → **8** | 10 → **6** | 7 → **3** |
| **total** | 40 → **35** | 31 → **27** | 15 → **9** |

The correction did not reverse the direction; it sharpened it. That is worth
stating plainly rather than quietly, because a correction that happens to
favour the hypothesis deserves more scrutiny than one that does not: the
false positives fell 5 on `control`, 4 on `advise` and 6 on `advise+audit`,
so the arm that gained most from the fix is the treatment arm. The reason is
mundane and checkable — the comprehension form is more common in the
scripts written after an audit — but anyone re-reading this should confirm
it rather than take it.

**A8 — the `advise+audit` arm confounds the intervention with a second
attempt, and this is not yet resolved.** Raised by Yusuf Ünlü. In
`advise+audit` the model sees its own script again and revises it; in
`control` and `advise` it answers once. Some part of the fall from 35 to 9
is mlcompass telling the model what is wrong, and some part is simply
getting a second go, and **this experiment cannot separate them.**

The fourth arm is now **built** and named `control+revise`: the control
prompt as its first turn, then a second turn saying only *review the script
you just wrote and fix any problems you find in it*. No findings, no
checklist, and no concern named — saying "check for leakage" would smuggle
in the content this arm exists to withhold, and the contrast would measure
prompt wording instead of mlcompass output. Its closing instruction is
byte-identical to the audit arm's, so the two second turns differ only in
the presence of the findings block. Two assertions run at execution time: the
first turn must equal the control prompt, and the second must contain no
mlcompass content.

`ARMS` is ordered so the tuple reads as the design, each neighbouring pair
isolating one thing:

| pair | what it isolates |
| --- | --- |
| `control` → `control+revise` | what a second attempt is worth on its own |
| `control` → `advise` | what the advice block is worth on one turn |
| `control+revise` → `advise+audit` | what mlcompass adds on top of a second attempt |
| `advise` → `advise+audit` | what the audit and its turn add to the advice |

Until the arm has **run**, the defensible claim stays as it was: **the arm
combining mlcompass findings with a revision turn produces fewer flagged
defects**, and not that the findings are what caused it. A built arm is not
a result, and the reports say no more than the data supports.

**A9 — 70 of 108 runs carry a held-out score; the report's conclusion is
weakened to match.** Also raised in the same review. 21 scripts failed to
run and 17 ran but saved a model the scorer could not load (see A6). The
shortfall is severe and uneven: `openai-gpt-5.4-mini` has **5 of 27** runs
scored, and **17 of the 36 cells have fewer than three scored repeats, 8 of
them none at all**. A "3 better, 3 worse, 3 flat" reading across cells is
therefore not resting on three repeats per cell.

"The process improves, the metric does not" claims more than this evidence
can carry. The claim in §6 stands as a pre-registered *prediction* that was
not refuted; it is not a measured null. Everywhere a result is reported it
now reads **no consistent improvement in held-out score could be shown**,
which is what 70 unevenly distributed scores support.

**A10 — the frozen plan was not enforced, and the harness's own instructions
caused the violation.** Found while running A8. `--experiment-id` checked
only that `plan.md` *existed* and then ran whatever the command line said.

Experiment `ab-20260915-795b90` froze **12 `control+revise` cells at 3
repetitions — 36 runs**. The execution did **48 runs across all four arms at
1 repetition**, appended them to `ab_results.csv` under the plan's own id,
and exited 0.

The cause is worth naming precisely, because it is not carelessness at the
keyboard. `--plan` printed:

```
Run it with:
  python benchmark/run_ab.py --experiment-id <id> --panel-id <p> ...
```

That line carries neither `--arm` nor `--repeats`. **Following the harness's
own instruction produced the mismatch.** A pre-registration whose tooling
hands you the command that breaks it is not a method; it is a paragraph.

Both halves are fixed. `read_plan_cells` parses the `## Cells` table back
into the arms, datasets, models, repetitions and seeds it commits to;
`enforce_plan` compares that against the run about to start and aborts,
naming every divergence, before anything executes or is written. The check
runs **before** `--dry-run` returns, because a violation you can only find by
spending money is not a check. The suggested command now carries every
selection flag, and `tests/test_ab_plan_enforcement.py` asserts both — the
parser against the real 795b90 plan on disk, and the refusal against the
exact combination that ran.

The 48 rows are **not deleted**. §7 keeps evidence, including evidence of our
own mistakes, and an experiment id that quietly disappears is worse than one
that is labelled. They carry experiment id
`ab-20260915-795b90-UNPLANNED` and a note stating what happened, so no
analysis can pick them up as the A8 result by accident. **A8 therefore
remains unrun**, and the claim it was built to test remains open.

**A11 — two more checker rules blind to a named constant, found the same
way and biased the same direction.** Running A8 produced what looked like a
finding: 3 of its 36 runs scored *worse* after the model revised its own
script, 33 unchanged, none better. Opening the three scripts, **two were the
checker and one was real.**

| run | flag that flipped | verdict |
| --- | --- | --- |
| `1480-…-deepseek-flash-r2` | `target_in_features` | **false positive** — `FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS`, then `X = df[FEATURE_COLS]`. The target is not in either list. |
| `1480-…-gpt-5.4-mini-r3` | `no_seed` | **false positive** — `RANDOM_STATE = 42`, passed in three places. The seed patterns want a digit. |
| `1464-…-gpt-5.4-mini-r2` | `no_validation` | **genuine** — no split, no cross-validation, nothing. |

The direction is the part that matters and it is not random. **A revised
script is a tidier script**, and tidier code hoists column lists and seeds to
module constants — exactly the forms a regex list misses. So the checker
systematically penalised the coding style that revision produces, which is
the style of the arm under study. Every rule scored by the *absence* of an
enumerated form carries this bias; A7 was the first instance, not the only
one.

Fixed: `_named_list_excludes_target` resolves a list of quoted names, and one
level of `A + B` concatenation, when the name is used in a `df[NAME]`
selection (form 8); `_int_constant_expansion` substitutes names bound to
integer literals before the seed patterns run. Both are narrow on purpose —
a comprehension is not a constant list, and a constant nobody selects with
proves nothing about the feature matrix. `tests/test_ab_target_exclusion.py`
grows to 24 cases, each accepted form paired with a script that genuinely
commits the defect.

Rescored as `ab-20260915-3b0221-rescored-v2`. The three scorings now read:

| scoring | control | advise | advise+audit |
| --- | ---: | ---: | ---: |
| original | 40 | 31 | 15 |
| rescored (A7) | 35 | 27 | 9 |
| **rescored-v2 (A7+A11)** | **33** | **26** | **8** |

The direction survives all three. It has now moved toward the hypothesis
three times running, and that is worth saying plainly rather than burying:
every correction so far has helped us. The mechanism is understood and
checkable — false positives concentrate in careful code, and the treatment
arms produce more careful code — but anyone reading this should confirm it
rather than accept it. **The next correction that moves the numbers the
other way will be the one that tells us the scorer is finally unbiased.**
