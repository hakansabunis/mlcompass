# A17 result: the rubric arm, and what it costs the paper

Run 2026-09-18 against frozen plan `ab-20260917-f98e3c`, 36 runs, 3 datasets ×
4 models × 3 repeats. Scored by the current `check_defects` (post-A16), which
is the same code the `v4` / `v4cr` rescores used, so the arms are comparable.

## The headline, and it goes against us

| arm | N | defects | per run |
|---|---|---|---|
| `control` | 36 | 31 | 0.86 |
| `control+revise` | 35 | 35 | 1.00 |
| `advise` | 36 | 24 | 0.67 |
| `advise+audit` | 36 | 8 | 0.22 |
| **`control+revise+rubric`** | **32 scored** | **1** | **0.03** |

The arm that is told the six scored concerns and **nothing about its own
script** ends at one flagged defect. That is better than the arm handed
`mlcompass audit`'s actual findings.

§9 A17 recorded, before this ran, that a result near 8 would mean "the fall is
largely the rubric and the manuscript's causal claim has to be rewritten". It
landed at 1. **The treatment–outcome vocabulary overlap is not a limitation to
note; it is a confound large enough to account for the whole effect on this
metric.**

## But the checklist is a proxy, and the arm broke the thing it stands for

| arm | ran | runnable | defects among runnable | per run |
|---|---|---|---|---|
| `control` | 28/36 | 78 % [62, 88] | 25 | 0.89 |
| `control+revise` | 29/35 | 83 % [67, 92] | 29 | 1.00 |
| `advise` | 29/36 | 81 % [65, 90] | 22 | 0.76 |
| `advise+audit` | 30/36 | 83 % [68, 92] | 7 | 0.23 |
| **`control+revise+rubric`** | **17/36** | **47 % [32, 63]** | 1 | 0.06 |

Every other arm executes between 78 % and 83 % of the time. The rubric arm
executes 47 %, and the intervals barely touch. Handed six conditions to
satisfy and no information about its own code, the model writes something that
satisfies the checklist and does not run.

This is Goodhart's law inside our own instrument, and we can measure it because
the benchmark records execution separately from the checklist. §12 already
argues that six deterministic rules are not code quality. This is the first
evidence in the study of what optimising against them actually costs.

## The outcome a practitioner gets

A run is useful only if it executes **and** carries no flagged defect.

| arm | useful | 95 % CI |
|---|---|---|
| `control` | 8/36 | 22 % [12, 38] |
| `control+revise` | 7/35 | 20 % [10, 36] |
| `advise` | 14/36 | 39 % [25, 55] |
| **`advise+audit`** | **23/36** | **64 % [48, 78]** |
| `control+revise+rubric` | 16/36 | 44 % [30, 60] |

On the joint outcome `advise+audit` is still first. The intervals overlap
substantially, so at N=36 this design separates it from the rubric arm
**weakly**, and we will not claim more than that.

## The finding that refines rather than removes a claim

§9 reports that self-revision repaired nothing: across 35 `control+revise`
pairs, **no rule flagged before the revision was clean after it**. That is
still true, and this arm explains it.

| arm | pairs | defects pre | post | runs improved | runs worsened |
|---|---|---|---|---|---|
| `control+revise` | 35 | 33 | 35 | **0** | 2 |
| `control+revise+rubric` | 32 | 25 | 1 | **21** | 0 |

Given a generic rubric and no findings at all, the same models on the same
datasets repair 21 of 25 defects and regress none. So "models cannot repair
their own code without external feedback" is the wrong reading of our own
result. The correct one is narrower: **a model asked to review its own script
with nothing named fixes nothing; a model told what to look for fixes most of
it, whether or not anyone tells it what is actually wrong.**

## What has to change in the manuscript

1. §9's heading `The extra attempt is worth nothing; the findings are worth
   31 → 8` cannot stand. The findings are not what is worth 31 → 8 on the
   checklist.
2. §12's internal-validity paragraph currently calls the overlap an objection
   "weakened but not removed" by `leak_duplicate_rows`. That is now too
   generous to us and must report this arm instead.
3. The `advise+audit` claim survives only on the joint outcome, with overlapping
   intervals, and has to be stated at that strength.
4. §12 gains the Goodhart finding, which is evidence *for* the paper's own
   position that a six-rule checklist is a proxy.

This is the fifth instrument-or-design correction in the study and the second
to move a result against the hypothesis we are arguing for.

## Reproducing

```bash
source scripts/load_keys.sh
python benchmark/run_ab.py --experiment-id ab-20260917-f98e3c \
  --dataset openml-1464 --dataset openml-1480 --dataset openml-44031 \
  --arm control+revise+rubric \
  --panel-id deepseek-flash --panel-id mistral-ministral-8b \
  --panel-id ollama-qwen2.5-7b --panel-id openai-gpt-5.4-mini \
  --repeats 3 --seed 0 --seed 1 --seed 2
```

Four of the 36 runs carry no defect count: two `no_code_revision`, one
`llm_failed_revision`, one `llm_failed`. Per §5 those are transport and
no-code outcomes, excluded from the scored denominator and reported here rather
than absorbed.
