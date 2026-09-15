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

Every figure below was checked on 2026-09-15 against the source linked
beside it. Two things move yearly and must be re-checked before you act:
quartiles, and article charges.

### Tier 1 — best fit, realistic odds

#### Empirical Software Engineering (EMSE) · Springer

| | |
| --- | --- |
| Metrics | **IF 3.6** (2026 release, on 2025 data; 3.4 the year before) · CiteScore 8.2 · **SJR Q1** |
| ISSN | 1382-3256 / 1573-7616 |
| Editors-in-chief | Robert Feldt, Thomas Zimmermann |
| Charges | **Hybrid.** Subscription route costs nothing. Open access is £2390 / $3390 / €2790, chosen *after* acceptance |
| Home | https://link.springer.com/journal/10664 |
| Submission guidelines | https://link.springer.com/journal/10664/submission-guidelines |
| Community site (special issues, RR) | https://emsejournal.github.io/ |

**Why first.** Our sharpest result is an empirical-software-engineering
result, not an AI result: a fabrication rate that moves from 1% to 100%
across six rule-free paraphrases of one instruction. EMSE is where a
measured claim about developer-facing tooling belongs, and its artifact
culture rewards what we actually have — a shipped MIT tool, green CI over
3.10–3.13 × two operating systems, 985 tests — instead of treating it as
decoration. The pre-registration discipline in `ab_protocol.md` reads as a
virtue here and as an oddity nearly everywhere else.

**One thing not to misread.** EMSE's Registered Reports track is
two-stage and **Stage 1 runs at a partner conference**, not at the journal
(https://emsejournal.github.io/registered_reports/). Our pre-registration
is our own discipline, not an EMSE RR submission, and it does not open a
shortcut. Cite it as method, not as status.

#### Journal of Systems and Software (JSS) · Elsevier

| | |
| --- | --- |
| Metrics | **IF 3.8** (2026) · **Q1** · CCF B |
| Charges | Hybrid; OA is **$3,850** ex-tax, subscription route free |
| Home | https://www.sciencedirect.com/journal/journal-of-systems-and-software |

**Why second.** Its scope statement is almost a description of this paper:
*all articles should provide evidence to support their claims, through
empirical studies, simulation, formal proofs or other types of
validation.* We have all three — the propositions, the live battery, the
benchmarks. JSS is also more comfortable than EMSE with a paper whose
centre of gravity is a working system. It ranks second only because our
measurement is stronger than our system, and EMSE weights the measurement.

#### Information and Software Technology (IST) · Elsevier

| | |
| --- | --- |
| Metrics | **IF 4.6** (released 17 June 2026, on 2025 data) · CiteScore 9.1 · **JCR Q1 and SJR Q1** |
| Charges | OA **$3,350–3,820** depending on the source consulted; confirm at submission |
| Home | https://www.sciencedirect.com/journal/information-and-software-technology |

Added to the list because it is the same family as JSS with a *higher*
impact factor, which surprises people who assume the ranking follows
prestige. Scope is improvement of software development practice. A
reasonable third shot, or a second if JSS declines.

### Tier 2 — higher ceiling, higher risk

#### ACM TOSEM

| | |
| --- | --- |
| Metrics | **IF 6.9** (2025 JCR) · 8th in Computer Science, Software Engineering |
| Charges | **Changed on 1 January 2026: ACM is now 100% open access.** An APC applies unless your institution is in ACM Open (2,700+ institutions). A **temporary 2026 subsidy** puts it at **$250 for ACM/SIG members, $350 for non-members** — a 65% discount, funded by ACM and not promised beyond 2026. Geographic and hardship waivers exist |
| Home | https://dl.acm.org/journal/tosem |
| Open-access terms | https://dl.acm.org/journal/tosem/open-access |
| APC list pricing | https://libraries.acm.org/acmopen/apc-list-pricing |

**Action item worth one email:** find out whether Istanbul Medipol
University is in ACM Open. The answer is the difference between a few
hundred dollars and a full APC, and the 2026 subsidy makes this the
cheapest year to publish with ACM that there has been or is likely to be.

#### IEEE TSE

| | |
| --- | --- |
| Metrics | **IF 6.5** (2023 JCR — the most recent figure I could confirm) |
| Review speed | First round **≈11.6 weeks** per author-reported data at https://scirev.org/journal/ieee-transactions-on-software-engineering/ |
| Home | https://www.computer.org/csdl/journal/ts · https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=32 |

**Both of these are reachable, and not with what is on disk today.** They
want the full grid — providers × tasks × leak patterns — plus the
head-to-head against Guardrails and NeMo. Our own harness currently marks
the NeMo arm *not built* and reports it as not-run rather than as a tie,
which is the honest call and also exactly the hole a TOSEM reviewer will
put a finger on. Attempt only after the baseline battery and the merge in
§1; budget 9–15 months.

Note: TOSEM's **Agentic AI in Software** special issue is **closed** —
submissions were due 1 Nov 2025, the issue published August 2026
(https://dl.acm.org/journal/tosem/agentic-ai-software). There is no open
agentic-AI special issue at TOSEM to aim at.

### Tier 3 — AI-side, fast, and a worse fit than the impact factor suggests

#### Expert Systems with Applications (ESWA) · Elsevier

| | |
| --- | --- |
| Metrics | **IF 9.4** · **Q1** |
| Charges | OA **$3,490** ex-tax |
| Speed | Medians: ~5 days to first decision (i.e. desk screen), 62 days to decision after review, 147 days submission→acceptance |
| Home | https://www.sciencedirect.com/journal/expert-systems-with-applications |

#### Knowledge-Based Systems (KBS) · Elsevier

| | |
| --- | --- |
| Metrics | Widely listed Q1; **I could not confirm a current 2026 impact factor** — the most recent figure I could verify is 8.038 from 2020. Check before quoting it |
| Home | https://www.sciencedirect.com/journal/knowledge-based-systems |

**Read the impact factors carefully.** 9.4 against EMSE's 3.6 does not
mean ESWA is three times the journal. Applied-AI venues carry far higher
citation density than software-engineering venues, so the numbers are not
comparable across categories — which is precisely why quartile, not raw
IF, is the thing to compare.

**And the fit is worse than the topic suggests.** These venues are built
around a novel method evaluated on benchmarks. Ours is a *contract* paper:
the mechanism is deliberately simple — set membership, a float comparison,
an anchor check — and the contribution is the framing plus the
measurement. That "none of the checks is sophisticated, and that is the
point" sentence in our own introduction is a strength at EMSE and a desk-
rejection risk at ESWA. The 5-day median first decision is mostly desk
screens.

If the goal is a Q1 line quickly, this is the shortest path. If the goal
is that the people building these tools read it, it is not.

### Not viable

| Venue | Why |
| --- | --- |
| INISTA 2026 | Closed 25 June 2026; conference is 17–19 Sep 2026 |
| TOSEM *Agentic AI in Software* SI | Closed 1 Nov 2025, published Aug 2026 |
| EMSE *Software Reliability* SI | Deadline was 1 March 2025 |

---

## 2b. One consequence of INISTA closing that is easy to miss

Because we never submitted to a conference, there is **no prior
publication to extend**. Springer, Elsevier, ACM and IEEE all require a
journal submission to be substantially new relative to any earlier
conference version — usually 30% or more. That requirement does not apply
to us. We submit the full paper directly, with no overlap to manage and
no self-plagiarism check to survive.

All four publishers permit an arXiv preprint before submission. IEEE
additionally requires a specific copyright notice to be added to the
preprint once a paper is accepted — check that wording if the work ends
up at TSE, and post to arXiv regardless, since the requirement applies
after acceptance and not before.

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
