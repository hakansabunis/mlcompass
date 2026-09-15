# Where to send this work

**Sweep date:** 2026-09-15 · **Manuscript:** `paper/inista_latex/main.tex` (head `51ad3c5`)

Everything with a date below was checked on the sweep date against the
venue's own page. Quartiles are Scimago unless stated; confirm on
scimagojr.com before committing, because they move yearly.

---

## 0. The thing that changed the plan

**INISTA 2026 is closed.** Submission closed 25 June 2026 — the page says
"strict deadline, no extensions will be granted" — and the conference runs
17–19 September 2026 in Guimarães, Portugal, i.e. two days from this sweep.
The manuscript is written to a 6-page IEEEtran conference format for a
venue that cannot receive it.

Two consequences, and the second is the useful one:

1. INISTA 2027 is the next cycle for that specific venue.
2. **The page limit is no longer binding.** The +1163 words the related-work
   correction added were a problem for a 6-page conference paper and are
   not a problem for a journal. Nothing needs to be cut. The manuscript
   should now be *extended*, not trimmed.

---

## 1. The honest constraint, before any venue list

Every Q1 software-engineering journal will stop at the same sentence, and
it is already in our own limitations section:

> **Single provider.** Every measurement comes from one commercial LLM.

That is the binding weakness, not the writing. The good news is we closed
it without noticing: the A/B benchmark ran **four model lanes** (deepseek,
mistral-ministral-8b, ollama-qwen2.5-7b, openai-gpt-5.4-mini) across 108
runs, and the detection benchmark added 45 more. Those numbers currently
live in `benchmark/` and are cited by no part of the manuscript.

**A journal version is not a rewrite. It is a merge.** The contract paper
supplies the mechanism and the guarantee; the benchmark work supplies the
multi-provider generalisation the contract paper is missing and the
independent finding that process improves while the metric does not.
Together they clear the bar that either clears alone.

Second gap, closing as this is written: the baseline comparison arms
(Guardrails AI stock, Guardrails running our Tier B validators, provider
strict mode with and without a stale enum) had never been run. The
battery is running now on the paper's own provider. Until it lands, the
manuscript's own limitation stands — "we ran no head-to-head against the
validate-and-reask toolkits" — and a Q1 reviewer will ask for exactly that
table.

---

## 2. Shortlist

### Tier 1 — best fit, realistic odds

**Empirical Software Engineering (EMSE)** · Springer · Scimago **Q1** ·
rolling submission, no special issue needed.

The closest match in the list, for one reason: EMSE is where a *measured
claim about developer-facing tooling* belongs, and our sharpest result is
exactly that shape. A fabrication rate moving from 1% to 100% across six
rule-free paraphrases of one instruction is an empirical-software-engineering
finding, not an AI finding. EMSE also has a strong artifact culture, which
rewards the thing we actually have — a shipped MIT tool, green CI across
3.10–3.13 × two operating systems, 985 tests — rather than treating it as
decoration. The pre-registration discipline in `ab_protocol.md` (frozen
plans, losing conditions written before data, an amendment log naming our
own two design defects) reads as a virtue here and as an oddity almost
everywhere else.

*What it needs from us:* the merge in §1, and the baseline table.

**Journal of Systems and Software (JSS)** · Elsevier · Scimago **Q1** ·
rolling.

Same family, faster in practice, and more comfortable with a paper whose
centre of gravity is a working system. Second choice rather than first only
because EMSE weights the measurement and JSS weights the system, and our
measurement is the stronger half.

### Tier 2 — higher ceiling, higher risk

**ACM TOSEM** and **IEEE TSE** · both **Q1**, both top-tier.

Reachable, but not with what is on disk today. These want the full grid —
providers × tasks × leak patterns — plus the head-to-head against
Guardrails and NeMo, plus a NeMo adapter that is currently marked *not
built* in our own harness and honestly reported as not-run rather than as
a tie. Worth attempting only after the baseline battery and the merge, and
expect 9–15 months.

Note: TOSEM's **Agentic AI in Software** special issue is **closed** —
submissions were due 1 Nov 2025 and the issue published August 2026. There
is no open agentic-AI special issue at TOSEM to target.

### Tier 3 — AI-side, faster turnaround

**Knowledge-Based Systems** and **Expert Systems with Applications** ·
Elsevier · both **Q1** by JCR and Scimago.

Genuinely Q1, genuinely faster, and they accept applied-AI work with an
evaluation without demanding the SE community's artifact apparatus. The
trade is reputational: inside software engineering these carry less weight
than EMSE or JSS for the same quartile. If the goal is a Q1 line on a CV
quickly, this is the shortest path. If the goal is that the work is read by
the people who build these tools, it is not.

### Tier 4 — open special issues that fit topically

**IEEE Software** — *Building Trustworthy Software in the Time of AI*
· **deadline 5 November 2026**, publication July/August 2027.

Verified open. The scope names testing and validation of AI components and
human oversight and fail-safe mechanisms, which is our paper's subject in
the magazine's own words. Caveats, both real: it is a **magazine**, not a
research journal (Q2, practitioner-facing, roughly 4700 words, few
equations, no room for the propositions), and it would consume the
material. Best treated as a *companion* to a journal submission, not a
substitute — the practitioner-facing version of the same argument, which
is also the version most likely to bring the tool users.

**Communications AI & Computing** (Nature portfolio, open access) —
*Safety, Trustworthiness and Robustness in Large Language Models*
· **deadline 31 May 2027**, confirmed open.

Topically a direct hit. One thing to be clear about before anyone gets
attached to the Nature branding: it is a **new journal with no impact
factor and no Scimago quartile yet**, so it cannot be counted as Q1 or Q2
today, whatever it becomes. It is also open access, so check the APC.

### Not viable

| Venue | Why |
| --- | --- |
| INISTA 2026 | Closed 25 June 2026; conference is 17–19 Sep 2026 |
| TOSEM *Agentic AI in Software* SI | Closed 1 Nov 2025, published Aug 2026 |
| EMSE *Software Reliability* SI | Deadline was 1 March 2025 |

---

## 3. Recommended sequence

1. **arXiv now**, with the corrected manuscript. Nothing blocks it, it
   establishes the date, and six concurrent works appeared in the last six
   weeks alone — three of them touching load-bearing claims. The cost of
   waiting is not zero.
2. **Finish the baseline battery**, merge the four-model benchmark evidence
   into the manuscript, and extend it to journal length. This is the work,
   and it is mostly writing rather than running.
3. **Submit to EMSE.** JSS if EMSE declines.
4. **Optionally, in parallel: IEEE Software by 5 November 2026.** Different
   audience, different register, no conflict with a journal submission of
   the research version — but confirm the magazine's prior-publication
   policy against the arXiv preprint before submitting.

The one thing not to do is hold the arXiv posting until the journal
decision. The field is moving in weeks.
