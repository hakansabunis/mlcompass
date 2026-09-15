# The baseline battery, and what it says the contract is actually doing

**Run:** 2026-09-15 · **Provider:** DeepSeek `deepseek-chat` ·
**Task:** synthetic, 10 evidence columns, anchor `log_target_v2` ·
**N = 200 per arm, 7 arms, 1,400 live responses** ·
**Evidence:** `scripts/runs/*.jsonl`, one record per response ·
**Harness commit:** `a77edad` · **Cost:** ~$0.51 estimated

This is the comparison block the manuscript's own limitations section said
was missing — *"we also ran no head-to-head against a static-schema decoder
where decoders apply, nor against the validate-and-reask toolkits."* The
arms had been written and never run. They have now run.

It produced two results. The first is the one we were looking for. The
second is not, and it is the more important of the two.

---

## 1. The comparison: it is the validator, not the loop

| Arm | What it is | Entity-fabrication | 95% CI | k/N |
| --- | --- | ---: | :---: | ---: |
| `A-L1` | bare prompt, no enforcement | **42.0%** | [35.37, 48.93] | 84/200 |
| `A-GR-STOCK` | **Guardrails AI**, stock validators + reask | **35.0%** | [28.73, 41.84] | 70/200 |
| `A-STRICT-STATIC` | provider strict mode, static schema, no enum | **4.0%** | [2.04, 7.69] | 8/200 |
| `A-CONTRACT` | **our contract**, bare prompt + Tier A + Tier B | **0.0%** | [0.00, 1.88] | 0/200 |
| `A-STRESS` | Tier B alone, no enum | **0.0%** | [0.00, 1.88] | 0/200 |
| `A-GR-OURS` | **Guardrails' loop, our Tier B validators** | **0.0%** | [0.00, 1.88] | 0/200 |
| `A-STATIC-ENUM-STALE` | strict mode, author-time enum | 0.0% | [0.00, 1.88] | 0/200 |

The two rows to read together are `A-GR-STOCK` and `A-GR-OURS`. **Same
toolkit, same reask loop, same budget of 2 — 35% against 0/200.** The only
thing that changed is what the validator checks. Guardrails' structural
validators confirm the answer has the right shape; they have no way to know
which column names the evidence contained, because nothing told them.

That is the paper's claim, measured against the nearest real alternative
rather than asserted: **the outer loop is general-purpose infrastructure and
was never the contribution. The validator inside it is.** Tier B is not a
better reask loop. It is a decidable check that a general-purpose loop
cannot express.

`A-STRESS` reaching 0/200 with the enum removed says the same thing from the
other side: post-generation verification alone is sufficient, which is what
makes the guarantee independent of the provider.

### One cell that must not be read as a result

`A-STATIC-ENUM-STALE` reads 0/200, and the harness itself refuses the
reading: on this task the author-time column list covers **10 of 10**
evidence columns, so the arm is byte-identical to a live enum *by
construction*. It is a degenerate cell. The arm is only informative on a
task the frozen list predates, and no such task is in this battery. The row
is printed for completeness and carries no comparison.

---

## 2. The result we were not looking for: nothing was invented

Every arm's flagged responses were decomposed into two kinds — a name that
appears **nowhere** in the evidence dictionary (an invention), and a name
that **is** in the evidence but was put in a field that does not hold
columns (a misfiling).

| Arm | flagged | misfiled | **invented** | what was cited |
| --- | ---: | ---: | ---: | --- |
| `A-L1` | 84 | 84 | **0** | `r2` ×87, `perfect_match_rate` ×1 |
| `A-GR-STOCK` | 70 | 70 | **0** | `r2` ×70 |
| `A-STRICT-STATIC` | 8 | 8 | **0** | `r2` ×8 |
| every other arm | 0 | 0 | **0** | — |

**Across 1,400 live responses on this task, the narrator invented nothing.**
Every single flagged entity is `r2` — the name of the suspicious metric,
which the evidence dictionary contains as `suspicious_metric.name` and which
the model wrote into `columns_referenced`.

This matters because of what the manuscript calls it. Its term is
**phantom-entity fabrication**, defined as *"a column the data does not
contain."* On this evidence that definition is not what is being counted. A
reviewer with these logs will say the rate measures a category error — the
model naming the metric it was handed — and not a hallucination, and on this
battery they will be right.

Three things follow, and none of them dissolves the contribution.

**It is still a contract violation.** `columns_referenced` means columns of
the data. A consumer acting on `r2` — dropping it, correlating it, listing
it as a leak candidate — is acting on something that does not exist as a
column. Tier B rejecting it is correct behaviour, not over-eager behaviour.

**The comparison in §1 is unaffected.** Every arm faced the same failure and
the arms still separate from 42% to 0/200. What the battery establishes is
that Tier B catches an out-of-domain name and Guardrails' stock validators
do not; that holds whatever the name's provenance.

**But the headline number changes meaning, and the paper's does not match
this one anyway.** The manuscript reports `A-L1` on this task at **11.5%,
CI [7.79, 16.66]**. Today, same task, same provider, same scorer: **42.0%,
CI [35.37, 48.93]** — far outside the published interval. The June records
are gone (`scripts/runs/` held none), so whether the original 23 were also
`r2` cannot be checked. Either the rate moved by a factor of four under a
rolling alias, or the two numbers are counting different things. Neither
possibility leaves the published point estimate usable as stated.

### What to do about it

Split the entity channel in the scorer, and report both:

- **invented** — a name absent from the evidence dictionary entirely. The
  hallucination the paper is named after. Measured here at **0/1,400**.
- **misfiled** — a real evidence name in a field that does not hold it. A
  contract violation, a real defect for a downstream consumer, and not a
  hallucination.

*"The bare narrator misfiles real evidence names at 42% and invents names at
0/1,400; Guardrails' stock validators catch neither; Tier B catches both"* is
a sharper claim than a blended rate, and it is one a hostile reviewer cannot
take apart, because it concedes the weaker reading up front.

### What is still unchecked

The **paraphrase sweep** — 1% to 100% across six rule-free rewordings — is
the manuscript's strongest single result and it has **not** been
re-decomposed this way. If those rates are also `r2`, the sweep measures how
wording changes a category error rather than a fabrication rate. The result
would survive in form, because the point is that wording moves the rate two
orders of magnitude; but what the rate *is* would have to be restated. This
is the next thing to run and it costs a rescore, not a battery.

---

## 3. The silent channels, and why the silence is not trivial

Value-fabrication and critical-omission were **0/200 in all seven arms**, as
in the published battery. The manuscript's limitation 2 names the way that
could be worthless: *"an arm that pushed the narrator into abstention would
trivially score zero on every channel."* It was never measured. It is now:

| | `A-L1` | `A-CONTRACT` |
| --- | ---: | ---: |
| abstained (`cannot_determine` or equivalent) | **2/200** | **0/200** |
| claims per response, median | **10** | **10** |
| never mentioned the anchor | **0/200** | **0/200** |

The narrator is not escaping into abstention. It commits, and it commits to
ten structured claims per response. **The zeros are not cheap**, and that
hole in the limitations section is closed with data rather than argument.

---

## 4. A provider defect found by running this

`--provider deepseek` had been scoring **every cell 0/0**. The default model
was repinned to `deepseek-v4-flash` on 2026-07-24 on a deprecation notice
and never exercised; every current DeepSeek direct name is a thinking-mode
model, and thinking mode refuses the forced `tool_choice` this harness
requires:

    400 - Thinking mode does not support this tool_choice

Probed with a forced `tool_choice`: `deepseek-chat` OK, `deepseek-flash`
400, `deepseek-v4-flash` 400, `deepseek-v4-pro` 400. `deepseek-chat` is
**absent from `models.list()` but still served**, so checking the listing
confirms the wrong conclusion.

The exclusion rule behaved exactly as designed: transport failures were
excluded rather than counted as clean non-fabricating responses, so the
tables read `0.0%` with `k/N` of **0/0** — a visible hole rather than a
silent lie. That rule exists because of the first defect Yusuf Ünlü found in
the detection benchmark. It earned its keep here.

Fixed in `a77edad`, with the probe recorded at the pin and the test rewritten
to assert the invariant that matters — that the default is a model the
contract can actually call — rather than a model name.

---

## 5. What this battery does and does not license

**It licenses:** that on this task, against the nearest real alternative,
the validate-and-reask loop is not what produces the guarantee — the same
loop with Tier B's validators goes from 35% to 0/200; that post-generation
verification alone suffices, so the guarantee does not depend on the
provider; and that the zeros on the value and omission channels are not an
artefact of abstention.

**It does not license:** a claim about *phantom* entities on this evidence,
because none were produced; the published 11.5% figure, which this run does
not reproduce; any claim about the stale-enum arm, which is degenerate here;
or generalisation beyond one provider, one task and one leak pattern.

**Next, in order:** re-decompose the paraphrase sweep (§2), split the entity
channel in the scorer and rescore the preserved records, then re-run this
battery on the real-data task where the stale-enum arm is not degenerate.
