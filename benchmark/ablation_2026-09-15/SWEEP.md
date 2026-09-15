# The paraphrase sweep, decomposed

**Run:** 2026-09-15 · **Provider:** DeepSeek `deepseek-chat` ·
**Task:** synthetic, 10 evidence columns, anchor `log_target_v2` ·
**6 variants × N=100 = 600 live responses, 0 transport errors** ·
**Evidence:** `scripts/runs/*sweep-*.jsonl` · **Cost:** ~$0.16

The sweep is the manuscript's strongest single result: a bare-prompt
fabrication rate moving across six rule-free rewordings of one instruction,
offered as the empirical case that prompt-level defences cannot be certified
from observed compliance. The baseline battery showed the blended entity
rate on this task is not what the paper calls it, so the sweep had to be
re-read the same way before the result could be restated.

It was worth running. The blended rate is not the whole story, and the part
it hides points the opposite way.

---

## The numbers

| Variant | Entity-fab (blended) | 95% CI | k/N | **invented** | misfiled | what was cited |
| --- | ---: | :---: | ---: | ---: | ---: | --- |
| terse | 9.0% | [4.81, 16.23] | 9/100 | **7** | 4 | `target` ×7, `r2` ×5 |
| baseline | 36.0% | [27.27, 45.76] | 36/100 | **0** | 36 | `r2` ×36 |
| expert | 57.0% | [47.22, 66.27] | 57/100 | **7** | 51 | `r2` ×54, `target` ×3 |
| cautious | 68.0% | [58.34, 76.33] | 68/100 | **2** | 68 | `r2` ×99, `target` ×2 |
| helpful | 71.0% | [61.46, 78.99] | 71/100 | **5** | 71 | `r2` ×112 |
| mechanical | 72.0% | [62.51, 79.86] | 72/100 | **1** | 71 | `r2` ×81, `row_count` ×3 |

- **invented** — a name that appears nowhere in the evidence dictionary. The
  phantom-entity fabrication the manuscript is named after. Verified: the
  frame's ten columns are `log_target_v2`, `near_target_proxy` and
  `feature_4` … `feature_12`. There is **no column called `target`**; a
  response citing `['log_target_v2', 'near_target_proxy', 'target']` invented
  the third.
- **misfiled** — a name that *is* in the evidence but is not a column,
  written into a field that holds only columns. `r2` is
  `suspicious_metric.name`; `perfect_match_rate` and `row_count` are evidence
  fields.

---

## 1. The paper's phantom claim is real, and an order of magnitude smaller

The baseline battery found **zero** inventions in 1,400 responses. The sweep
finds them: **0 to 7 per 100**, in five of six variants. So
phantom-entity fabrication is not an artefact of the scorer and the
manuscript is not describing something that never happens.

But it happens at **1–7%**, not at the 36–72% the blended rate reports.
Every headline figure in the manuscript that is offered as a *fabrication*
rate is, on this evidence, dominated by misfiling.

Both remain contract violations — a consumer that drops or correlates `r2`
acts on a non-column — and Tier B rejects both identically. What changes is
what the number may be called.

## 2. The two channels do not move together

This is the finding that was not expected, and it is the more interesting
one.

```
variant      blended   invented
terse            9%          7     <- lowest blended, highest invented
baseline        36%          0     <- zero inventions at 36%
expert          57%          7
cautious        68%          2
helpful         71%          5
mechanical      72%          1     <- highest blended, near-zero invented
```

**The blended rate and the invention rate are close to unrelated across
wordings, and at the extremes they run opposite.** The terse instruction
produces the fewest flagged responses overall and the most invented columns.
The mechanical field-listing instruction produces the most flagged responses
and almost no inventions — it drives the model to enumerate evidence fields,
which is misfiling, not invention.

The manuscript's thesis survives this intact, and arguably lands harder. Its
claim is that a prompt-level defence cannot be certified from observed
compliance. The decomposition shows you cannot even certify one failure mode
from another *within the same prompt*: a wording that looks safest on the
headline metric is the one inventing most.

## 3. The published range does not reproduce

The manuscript reports the sweep as **1% to 100%**. Today, same task, same
provider, same six variants, same scorer: **9% to 72%**.

This is the second figure in a day that does not reproduce; the `A-L1` cell
went 11.5% → 42.0% on the same run. Both were measured against a rolling
model alias and the June records did not survive, so the two cannot be
reconciled — only reported.

What survives is the qualitative claim, and it is the claim the paper
actually rests on: **rule-free rewording moves the rate roughly eightfold
(9% → 72%), across variants written before any measurement and none
discarded afterwards.** A defence that moves that far under innocuous
paraphrase cannot be certified from observed behaviour. The specific span
must be restated from this run, with its date and model pinned.

## 4. Abstention is not what produces the low cells

Checked, because a variant that pushed the model into abstention would score
a low rate for free: `terse` 1/100, `baseline` 2/100, `cautious` 0/100,
`expert` 1/100, `helpful` 0/100, `mechanical` 8/100. Even `mechanical`, the
highest, is 8%. The low cells are low because the model cited cleanly, not
because it declined to answer.

---

## What to change in the manuscript

1. **Report both channels wherever a rate appears.** "Invents columns at
   0–7% and misfiles real evidence names at 4–71%" is narrower than the
   current claim and is the claim the data supports.
2. **Restate the sweep's span** as 9%–72% from this run, with the date and
   the model pinned, and say plainly that the earlier span was measured
   against an alias whose records did not survive.
3. **Add §2 as a result.** That the two channels move independently across
   wordings is new, it is directly on the paper's thesis, and it costs one
   table.
4. **Keep the conclusion.** Nothing here weakens *prompts are advisory,
   schemas with verification are enforcement*. The contract rejects both
   channels at 0/200 in every arm of the battery, and it does not care which
   kind of name it is rejecting.
