# Hostile review — `paper/INISTA_Paper.md`

**Reviewer stance:** hostile but fair, calibrated to TSE / TOSEM / ICSE.
**Date:** 2026-09-14
**Materials read:** full manuscript; `paper/related_work_positioning.md`;
`paper/references_new.bib`; `paper/analysis_plan_2026-09.md`;
`paper/analysis_plan.md` (v1.0); the four `paper/ablation_live_*.md` run
records; `src/mlcompass/agents/leakage_investigator.py`;
`src/mlcompass/ui/evaluate.py` (pre- and post-`96f0071`); `src/mlcompass/cli.py`;
`src/mlcompass/mcp_server.py`; `src/mlcompass/agents/evaluate.py`;
`scripts/independent_scorer.py`; `scripts/reproduce_hallucination_ablation.py`;
`.gitignore`; git history.

Everything I assert about the artifact below was checked against the repository
at `96f0071`. Where I quote the manuscript I quote it exactly.

---

## Summary verdict

**For TSE / TOSEM / ICSE: reject as submitted.** Not because the idea is weak —
it is not, and §"What this paper does better than most submissions" says so at
length — but because at that tier a paper is rejected on any *one* of the
following, and this manuscript currently has all five:

1. A proposition presented as the paper's own that is a 2005 result in a
   neighbouring field (F1).
2. A guarantee sentence in the methodology section that is **false as written**
   about the paper's own system (F2).
3. A factual claim about the artifact that was **false when written** and is
   load-bearing for the paper's treatment of its largest unverified channel
   (F3).
4. A headline table whose zeros have an **admitted, unresolved confound** and
   whose raw data is **not in the repository**, so the confound cannot be
   resolved by a reviewer (F4, F5).
5. **No baseline.** Zero comparison against Guardrails, NeMo, a static strict
   schema, or a decode-enforced enum — in a paper whose contribution is an
   enforcement mechanism (F9).

**For TOSEM / TSE with a major-revision route:** major revision is achievable,
but only if F1–F9 are all addressed, and F9 requires the Round-2 baseline block
in `analysis_plan_2026-09.md` §2 to have actually run. That is months, not
weeks.

**For INISTA (the actual named target):** accept after revision, conditional on
F1, F2, F3, F6, F10 and F11. These are all fixable in the text without new
measurements. F1 in particular converts the paper's most vulnerable claim into
one of its better ones. Do not submit anywhere with F2 and F3 unfixed; those two
are the difference between "a paper that overclaimed" and "a paper that made a
false statement about its own artifact", and reviewers treat those differently.

**The single worst outcome available to you** is that a reviewer with a database
background reads Prop. 2, a reviewer doing artifact evaluation clones the repo,
and they compare notes. F1 plus F5 plus F14 in one review is unrecoverable at
any venue.

---

## Tier 1 — findings likely to sink the submission

### F1. Proposition 2 is Chomicki & Marcinkowski 2005, presented as the paper's own

**Target.** Abstract: *"Two short formal observations carry the design:
verification is linear-time, and deletion repairs soundness safely while no safe
repair exists for completeness."* Contribution 1: *"the two formal facts the
design rests on"*. And Prop. 2 itself, stated with a **Proof** and a ∎, with no
citation of any kind.

**Why a reviewer objects.** The universal/existential repair asymmetry —
constraints of the form "nothing unlicensed may be present" are repairable by
deletion, constraints of the form "something must be present" are not — is
*Minimal-change integrity maintenance using tuple deletions*, Information and
Computation 197(1–2):90–121, 2005. It is a canonical result. It is also AGM
contraction-versus-expansion in belief revision. Presenting it under a ∎ as one
of "the two formal facts the design rests on" is not a citation slip; it reads
as a priority claim. A database-theory reviewer will say so in the first
paragraph of their review and nothing after that paragraph will be read
charitably.

It is worse than a missing citation because **the paper's framing depends on the
proposition being non-obvious**: "the asymmetry looks like a preference and is
not. It is forced, and we record why in two propositions." If the asymmetry is
textbook, then "we record why" is the wrong verb.

Two aggravating factors. (a) `arXiv:2609.04343` (3 Sep 2026) uses the
manuscript's exact unsoundness/incompleteness vocabulary and takes a
removal-based repair — two weeks old, and a reviewer will assume you missed it.
(b) FAX (`arXiv:2605.27879`) already *does* retain/remove/omit — the strip-vs-flag
behaviour — without formalising it. So you cannot claim the observation either.

**What to do.** Exactly what the positioning package recommends, and do not
soften it further. Demote Prop. 2 from a proposition-with-proof to a *transposition
with a safety argument*. Cite `chomicki2005minimal` inside the statement. Then
state the part that is genuinely yours and state it as the whole point:

> In a database there is no speaker, so insertion is merely a different repair
> operation with a different cost. In narration, insertion **fabricates
> attribution**: it puts words in the narrator's mouth, which is the exact
> failure the contract exists to prevent. Deletion is therefore not the cheaper
> repair but the only *admissible* one. The corollary has no database analogue
> either: substituting an abstention is also inadmissible, because it attributes
> an epistemic stance the model did not take.

That corollary, as far as the sweep could establish, is unstated anywhere. It is
small. It is yours. Say it is small and yours, and let Prop. 1 and the
measurements carry the weight. Contribution 1 must be reworded in the same pass —
"the two formal facts the design rests on" currently claims both.

---

### F2. The §III-C invariant is false about your own system

**Target.** §III-C: *"The invariant that results is easy to state: **any response
the contract returns cites only entities present in the deterministic evidence
and quotes only values equal to it, by construction, whoever the provider is.**"*

**Why a reviewer objects.** This is false, and the paper knows it is false,
because the Introduction states the correct, scoped version four pages earlier:
*"every entity and every number in the validated citation and claim channels is
present in, and equal to, the deterministic evidence"* (emphasis on *in the
validated citation and claim channels* — mine, and it is doing all the work).

The response the contract returns is not the citation and claim channels. It is
the tool payload, whose schema (`build_submit_investigation_tool`,
`leakage_investigator.py` ~line 361) is:

```
"required": ["verdict", "confidence", "columns_referenced", "claims", "narration"]
```

`narration` is a required free-text string. `recommended_checks` is a free-text
array. Tier B never reads either. A response containing the sentence
"`phantom_col` correlates at 0.42 with the target" in `narration`, and no
corresponding entry in `claims`, passes every Tier B check and is returned by
the contract. So "any response the contract returns … quotes only values equal
to it" is not true of any response the contract returns.

This is not a pedantic reading. The system prompt's rule 2 —
*"Every correlation number you mention MUST also be reported in `claims`"* —
is exactly the rule that would close the gap, and it is **enforced by nothing**.
Tier B validates the claims that were made; it never checks whether a number in
`narration` was declared as a claim. That is a prompt-level defence, inside a
paper whose thesis is *"Prompts are advisory; schemas, with verification, are
enforcement."* A reviewer will quote your own slogan back at you.

**What to do.** Delete the unscoped sentence or scope it to the two channels,
matching §I. Then — and this is the part that turns the finding into a
contribution — state the residual explicitly as a *named, measured* gap:
the contract's guarantee covers the structured channels; the free-text channel
is unguarded by construction; the undeclared-number path is a live escape from
the guarantee that the system prompt asks for and the verifier does not check.
A "declared-number completeness" check (scan `narration` for numeric literals,
require each to appear in `claims`) is a half-day of work and would be a genuine
strengthening. Either ship it, or name it as the sharpest open item in §V-E.
Do not leave the invariant sentence as written.

---

### F3. §III-B and Limitation 4 make a false factual claim about the artifact

**Target.** Two sentences, both false when written.

§I: *"Free-text narration sits outside these channels, and **renderers surface
the validated fields as the actionable output.**"*

§V-E Limitation 4: *"nor is the free-text narration field (**renderers surface
the validated citation and claim sets as the actionable output**); enforcement
pressure could push fabrication into that unverified channel, and we did not
measure such displacement."*

**Why a reviewer objects.** The renderer, at the time those sentences were
written (`render_leakage_narration`, `src/mlcompass/ui/evaluate.py` at `96f0071^`),
emitted in this order:

1. Verdict / confidence
2. `**Hypothesis:** <primary_hypothesis>` — the **unverified free text**
3. `**Evidence cited:**` — the validated entity set
4. `**Recommended manual checks:**` — unverified free text

The unverified prose was the panel headline, above the verified findings, in
undimmed body style identical to the verified section. And `claims` — the
validated *value* channel, the one the paper's Tier B (2) exists to produce —
**was not rendered at all**. So the sentence is false twice over: the validated
fields were not surfaced as the actionable output, and one of the two validated
fields was not surfaced at all.

This matters more than a typical factual slip because of what the sentence is
*doing* in the argument. Limitation 4 is where the paper discharges its largest
known hole — the unverified free-text channel. Its entire mitigation is this
parenthesis. The mitigation was a claim about the renderer rather than a check,
and the claim was untrue. Strip it out and Limitation 4 reads: "the free-text
channel is unverified, we did not measure displacement into it, and we have no
mitigation." That is the honest state.

A reviewer doing artifact evaluation runs `mlcompass evaluate --llm` on a leaky
frame and sees the panel. This is a ten-minute discovery.

**What to do.** Three things, in order.

1. The renderer is now fixed (`96f0071`: verified content first, prose dimmed
   and labelled `model prose — NOT verified`). Good. But the paper must not
   simply re-assert the fixed state as if it were always true. State the
   *design principle* in §III-B — "the contract's output is partitioned into
   verified and unverified regions, and the renderer must present the partition"
   — and then say the renderer enforces it, with the commit as evidence.
2. Rewrite Limitation 4 so the mitigation is a *check*, not a *claim*. Right
   now nothing in the codebase tests the ordering. Add a renderer test asserting
   that the verified block precedes the prose block and that the prose carries
   the unverified label, and cite that test. A limitation discharged by a test
   is a limitation discharged; one discharged by a sentence is not.
3. Consider disclosing that the renderer did not originally do this and that
   you found it in your own audit. Reviewers reward that far more than they
   punish it, and the git history is public either way.

**Still open after the fix, and the paper should say so.** `render_leakage_narration`
still never renders `omitted_critical_evidence` or `had_unrecoverable_violation`.
Both are returned by `investigate_leakage_bound` (lines 747–748) and are read by
**no code anywhere in `src/`** — I grepped. So:

- Prop. 2's whole consequence is that completeness can only be *"detect,
  re-prompt, and flag"*. §III-C: *"a persistent omission is **flagged**"*.
  Flagged to whom? The flag reaches a dict key and stops. The one channel where
  the paper's own theory says the user must be warned is the one channel with no
  user-facing surface.
- Likewise a stripped phantom (`had_unrecoverable_violation`) is removed
  silently. Defensible — stripping *is* the repair — but the user has no signal
  that the narrator misbehaved, and §IV's telemetry story implies otherwise.

Fix the renderer for both, or state plainly in §V-E that the flag is telemetry
and not yet a user-facing surface. Do not let a reviewer find that the paper's
Proposition-2-mandated mechanism is unwired.

---

### F4. The 0/200 cells have an admitted abstention confound, and the paper leans on them anyway

**Target.** Table I rows L2 and L3; contribution 2 (*"the contract held the
user-facing rate at 0/200 on all three channels of both tasks"*); the abstract
(*"held the user-facing rate at 0/200 on all three channels"*).

Against §V-E Limitation 2: *"We also did not log the composition of L1
violations, per-response claim counts, **or per-arm abstention rates; an arm that
pushed the narrator into abstention would trivially score zero on every
channel.**"*

**Why a reviewer objects.** This is not a minor logging gap; it is the exact
alternative explanation for the headline result, conceded in the paper, and left
unresolved. Three facts compose:

- L2's strict prompt (§III-C Layer 2) explicitly instructs: *"abstain
  (`cannot_determine`) instead of guessing."*
- The completeness check exempts abstentions by design: *"an explicit abstention
  is not an omission"* (§III-C; `leakage_investigator.py` ~line 650).
- Entity and value fabrication both require the narrator to cite or claim
  something. An abstaining narrator cites nothing.

So a configuration in which the strict prompt drove the model to
`cannot_determine` at high rates produces **0/200 on all three channels
simultaneously** — exactly the observed result — and is indistinguishable from
the result the paper reports. Your own Round-2 plan makes abstention a
gating criterion for precisely this reason (§4.3: *"Abstention below 80% in every
cell counted… A cell at or above 80% abstention is unevaluable and licenses
nothing; the zero there is trivial"*).

The paper's counter-evidence is one sentence: *"L1's high rates indicate the task
elicits committed, citing behaviour, but the logging gap is real."* L1 is a
different prompt. It says nothing about L2/L3 abstention.

**Aggravating factor (see F5): the data to settle this is not in the repository.**
Abstention is trivially recomputable from raw per-response verdicts — if the raw
per-response records existed. They do not.

**What to do.** Recompute abstention rates from the raw records if you still hold
them anywhere. If you do not, say so in the limitation and say it is
unrecoverable for the June battery, then make abstention a reported column in
every table from Round 2 onward, as the plan already requires. Do not leave the
abstract asserting a three-channel zero whose most obvious alternative
explanation your own limitations section names and cannot rule out.

---

### F5. "Raw live-run records are in the repository" — they are not

**Target.** §VI, final line: *"All prompts, seeds, the measurement harness, and
raw live-run records are in the repository."*

**Why a reviewer objects.** Checked at `96f0071`:

- `scripts/runs/` contains exactly **one** file: `evidence_618bd74a.json`, an
  evidence dump. There are **zero** `.jsonl` files in the repository outside
  `.venv/`.
- The harness writes its per-response log to `scripts/runs/*.jsonl`
  (`reproduce_hallucination_ablation.py` line 1249). None exist.
- `.gitignore` carries a blanket `runs/` under "ML artifacts" and then attempts
  `!scripts/runs/`. Git does not re-include files under an excluded directory,
  so the negation is inert. The one tracked file must have been force-added.
- `scripts/independent_scorer.py` — which `analysis_plan_2026-09.md` §1.2 lists
  as **support item 4 for the mechanism claim C-S**, and §6 makes the sole
  source of manuscript tables — takes `--log runs/x.jsonl`. It has nothing to
  read for any published cell.
- The only run artifacts for the June battery are the four hand-written Markdown
  summaries in `paper/`. Those are *reports*, not records.

So the sentence is false, and it is false about the specific thing artifact
evaluation checks first. Worse, it is false in a way that makes F4 unfixable:
a reviewer who wants to check abstention, or claim counts, or violation
composition, or re-score with your own independent scorer, cannot.

**Compounding: the one commit pin in the run records does not resolve.**
`paper/ablation_live_deepseek_2026-06-12_battery.md` line 5:
*"**Harness:** `scripts/reproduce_hallucination_ablation.py` @ commit 17a2766"*.
`git cat-file -t 17a2766` → `fatal: Not a valid object name`. The other three run
records carry no pin at all. Set against §IV-A's reproducibility argument —
*"the name is a rolling alias, so run dates and the harness commit pin the
configuration"* — the pin is the whole mechanism, and it is dangling. Your own
Round-2 plan (§1.5) notes the `deepseek-chat` alias *"was deprecated within about
six weeks of the run"*. So: unreproducible configuration, dangling pin, absent
raw records. Three independent failures of the same property.

**What to do.** Either commit the raw JSONL (fix the `.gitignore` — move
`scripts/runs/` out from under the `runs/` rule entirely rather than relying on a
negation git will not honour), or change the sentence to state exactly what is
there. If the June JSONL no longer exists, say that in §IV, in one plain
sentence, and re-run one cell with logging on as a provenance anchor. Do not
ship the current sentence.

---

### F6. "Production system" + four wrong statistics in the same package

**Target.** Abstract: *"The contract ships in the production narrator of
mlcompass, an open-source ML-pipeline assistant."* §I: *"The contract is not a
prototype. It runs in the production narrator of **mlcompass**"*. §IV:
*"Everything above is the shipped code path, not a research fork."* §VI:
*"**Source code:** github.com/hakansabunis/mlcompass"*.

**Why a reviewer objects.** You are pointing reviewers at the repository as
evidence of production quality. At that repository, commit `96f0071`, dated after
the manuscript, is titled *"Fix four wrong statistics and stop advise from
inventing numbers"*, and its message documents:

- the drift chi-square was a goodness-of-fit test where a homogeneity test was
  required, *"rejected at about 28% for a nominal 5% test"* against two samples
  from the same distribution, with small-cell handling whose own comment
  described a merge it did not perform;
- binary AUC ignored ties, was **row-order dependent**, and returned 1.0 or 0.0
  for a constant-probability model whose true AUC is 0.5;
- PSI severity had no finite-sample correction — two 250-row halves of the *same*
  frame produced *"retraining is recommended"* 78% of the time;
- date columns were typed by dtype alone in the drift monitor, so a
  forward-moving `signup_date` scored PSI 14.82 "major" *"in the same row that
  reported a p-value of 1.0000"*;
- and `advise` printed a model-invented expected-AUC range in the same table
  style as measured statistics, with no provenance marker, persisted to disk.

The previous commit, `169187b`, is *"Fix the KS p-value, a boolean
hyperparameter, and an overclaimed README"*.

A reviewer will draw two conclusions and both are fair. First, "production" and
"not a prototype" are load-bearing rhetorical claims that the artifact does not
support. Second — and this is the one that actually damages the paper —
**the AUC bug manufactured the paper's own trigger condition.** §III-C:
*"which fires when an evaluation sees a metric that should not happen (AUC >
0.995, accuracy > 0.99, or R² > 0.999)"*. The commit message states it directly:
*"a spurious 1.0 trips the automatic leakage investigation."* The system that
exists to stop an LLM inventing a leak was being invoked by a deterministic bug
inventing one.

**What to do.** Drop "production" as a rhetorical device; it buys you nothing a
reviewer will grant and costs you everything if they clone. Replace with the
claim you can defend and that is actually stronger for an SE venue: *the contract
is the shipped code path of an open-source tool, measured in place rather than in
a research fork, and the same audit discipline was applied to the deterministic
layer.* Then disclose the four fixes in §V-E — not as an embarrassment but as
the evidence for F7 below. A paper that audits its own trusted layer, finds four
defects, fixes them, and reports it is doing something almost no submission does.
A paper that says "production" and gets caught is doing the opposite.

---

### F7. The trust boundary is asserted, not argued — and the deterministic side was wrong four times

**Target.** §III-B: *"A dashed trust boundary splits the picture in two. On one
side is everything deterministic… all plain code. On the other side is the *LLM
narrator*, which we treat as untrusted no matter who serves it."* §III-A's
out-of-scope list: *"(i) adversarial column names… (ii) provider outages; (iii)
miscalibrated confidence on evidence that is cited and valued correctly."*

**Why a reviewer objects.** "Deterministic" is not "correct". The paper's entire
architecture rests on a binary — trusted plain code, untrusted model — and
never argues for the first half. The threat model lists three exclusions and
*"the evidence producer is wrong"* is not among them. The closest the paper comes
is Limitation 4's *"if the deterministic layer misses the true leak"* — a
**recall** caveat. There is no **soundness** caveat for E at all.

The guarantee is, read precisely, a *consistency* guarantee, not a *truth*
guarantee: every number the user sees equals the number the deterministic layer
measured. If the deterministic layer measured it wrong, the contract propagates
the wrong number with a verification stamp on it, and the renderer now labels it
`verified against the evidence`. F6 supplies four demonstrated instances of the
deterministic layer being wrong in the same package. A reviewer will connect
these two facts and it will be the most interesting paragraph of their review.

**What to do.** Do not defend the boundary; *refine* it, and take the credit.
Add a short subsection (or fold into §III-A) stating:

- The guarantee is evidence-consistency, not truth. Say it in those words, once,
  plainly. It costs one sentence and pre-empts the objection entirely.
- E's correctness is a separate obligation discharged by a separate method —
  unit tests, property tests, statistical review — and **report that you ran
  exactly such an audit and it found four defects**. That converts F6 from a
  liability into the paper's answer to F7.
- The composition risk is the interesting one and nobody has written it down:
  a faithfulness contract over a wrong evidence producer **launders** the error.
  Pre-contract, a user might discount a strange number as model noise.
  Post-contract, it carries a verification badge. That is a genuine, novel,
  slightly uncomfortable observation about evidence-bound narration, and it is
  free — you already have the material.

Also: III-A's *"well-behaved provider"* directly contradicts §I's *"trusting no
provider enforcement"*, §II's *"enforcement never leaves our own code"*, and
§III-B's *"untrusted no matter who serves it"*. Pick one. The defensible split is:
*honest operator, untrusted narrator, provider assumed available but not assumed
to enforce anything*.

---

### F8. "The guarded path is the only path from narrator to user" is false on two shipped surfaces

**Target.** §III-B: *"Inside mlcompass the boundary sits within a larger system:
eleven pipeline tools behind a CLI, a tool-integration server [7], an autonomous
agent, and editor slash commands… **Every one of those surfaces routes through
the same deterministic backend, so the guarded path is the only path from
narrator to user.**"*

**Why a reviewer objects.** Routing through the same deterministic *backend* is
not routing through the *contract*, and the sentence equivocates between them.
Two concrete counterexamples in the shipped code:

1. **The MCP surface has no contract at all.** `mcp_server.py`'s
   `mlcompass_evaluate` returns `_json_safe(result)` — the full evaluation dict
   **including `leakage_investigation`**, i.e. the evidence dictionary E. The
   narrator on that surface is the *client's* LLM (Claude Desktop, Cursor,
   whatever). Tier A never runs. Tier B never runs. The evidence goes to an
   unguarded narrator and from there to the user. This is precisely the
   configuration the paper exists to prevent, shipped as a first-class surface,
   and it is the surface cited to [7] one clause earlier.

2. **The CLI runs a second, unguarded narrator over the same evidence, first.**
   `cli.py` lines ~974–988: `_maybe_interpret_evaluation(result, …)` runs and is
   rendered *before* the guarded leakage panel. It calls
   `agents/evaluate.py::interpret_evaluation`, which is `Agent(system=…,
   tools=[], …)` — a pure prose reasoner with no enum, no verification, only a
   `required_keys` shape check — and it is handed `result`, which **contains
   `leakage_investigation`**. So in the same terminal output, above the contract
   panel, an unverified narrator comments on the same evidence.

**What to do.** Delete the sentence and replace it with the true and still
useful one: *the contract governs the `evaluate --llm` leakage-investigation
path; other narration surfaces in the same tool are currently unguarded, and
extending the contract to them is engineering, not research.* Then say which
surfaces, and say that the MCP case is structurally harder because the narrator
is the client's and you control neither its schema nor its decoding — which, note,
is *an excellent motivating example for your own thesis* and currently wasted.

---

## Tier 2 — major findings, each individually revision-forcing

### F9. No baselines. None.

**Target.** §V-E Limitation 5: *"We also ran no head-to-head against a
static-schema decoder where decoders apply, nor against the validate-and-reask
toolkits [18], [19]."*

**Why a reviewer objects.** At ICSE/TSE, a paper whose contribution is an
enforcement mechanism and which compares against nothing is not evaluated, it is
described. The manuscript names its two closest practical relatives in §II —
*"The closest practical relatives are the validate-and-reask toolkits Guardrails
AI [18] and NeMo Guardrails [19], which ship the same outer loop as
general-purpose infrastructure"* — concedes the loop is theirs, and then does not
run them. A reviewer's question writes itself: if the loop is theirs and the
validator is yours, what does a Guardrails loop carrying your checks do? Your own
Round-2 plan has that arm (`A-GR-OURS`, H11) and pre-registers the losing
condition (LC-5: *"the verifier is the contribution and the loop is
interchangeable"*). You know. The paper does not admit that you know.

Second missing baseline, and it is the one that hurts: **a static strict schema
and a decode-enforced call-time enum**. `investigate_leakage_bound` already has
`strict_tools`. The plan's H9 *pre-concedes the tie* on the entity channel. So
the paper's single measured channel is one your own pre-registration expects a
simpler mechanism to match.

**What to do.** For INISTA: state the absence in §I as a scope limit, not only in
§V-E, and state the pre-registered expectation (H9's predicted tie) so no
reviewer can claim you were hiding it. For a journal version, the baseline block
is not optional.

### F10. The manuscript overclaims relative to its own pre-registered analysis plan, in four specific places

The plan draws a clean three-way split. The manuscript predates it and does not
honour it. Point by point:

**(a) "All three channels" collapses C-M and C-S.** Abstract and contribution 2:
*"the contract held the user-facing rate at 0/200 on all three channels of both
tasks"*. Per plan §1.3, entity and value zeros are *implementation-consistency
checks on an invariant that holds by construction*; the **omission** zero is *"an
ordinary empirical rate carrying all the weakness a 0/200 has"*. The manuscript
presents all three identically. Fix with the plan's own frozen §1.4 wording,
which does not appear in the manuscript at all.

**(b) "Prevented" is used for a channel where Prop. 2 forbids it.** Abstract:
*"fabrication prevented outright rather than statistically reduced"*; §VI:
*"fabrication on the verified channels is therefore preventable rather than
merely mitigable."* Plan §4.5, last row: omission events license *"detected and
surfaced"* and explicitly **not** *"prevented" (Prop. 2 forbids the word)*. The
abstract's "all three channels" sentence sits two clauses from "prevented
outright". A reviewer reading Prop. 2 and then the abstract sees the
contradiction without help.

**(c) §V-C claims boundary status the plan says is unlicensed.** Section heading:
*"### C. Negative Results Are Boundaries, Not Failures"*, and §IV-C: *"The other
two channels stayed silent everywhere."* Plan §4.6 licenses a persistent zero as
a **publishable boundary** only after the full §4.3 battery: derived-value
pressure, crowded/near-tie pressure, anchor-salience pressure, the `mechanical`
worst wording, ≥2 providers including open-weights, abstention < 80% in every
cell. *None of that ran.* Plan §4.3's closing line is unambiguous: *"If any
element is missing, the outcome is 'not yet measured', not a boundary."* The
correct current label is "not yet measured". The section as written claims the
Round-2 deliverable before Round 2.

**(d) §V-B makes a causal attribution the pre-registered control was built to
test, and the control was never run.** §V-B: *"The verifier does more than block:
it names what went wrong, and **that named feedback is what cut the
repeat-violation rate from 37.5% to 1.3%**."* `analysis_plan.md` v1.0 A3.2
introduces the generic-retry control precisely to separate naming from merely
retrying, and the shipped code implements it —
`correction_style="generic"`, documented in the docstring as *"the preregistered
H5 control that separates the effect of NAMING the violation from the effect of
merely getting a second attempt."* No run record contains a generic arm; I
grepped all four. So the paper asserts the causal conclusion its own
pre-registration says is not established, using a control that exists in the
repository and was not exercised. A reviewer who reads the code — and at ICSE
they do — finds the control before they find the result.

Weaken to: *"a corrective retry carrying the violation cut repeat violations from
37.5% to 1.3%; whether the effect is the naming or the retry is not separated
here, and the generic-retry control is registered for the next round."* That
sentence is honest, still striking, and costs you nothing.

### F11. Two of the four novelty conjuncts are occupied, and the paragraph undermines itself

**Target.** §II: *"To our knowledge, no prior work packages evidence-bound
domains, claim-level value checks, completeness checking, and provider-independent
enforcement as a single contract over a decidable faithfulness class."*

**Why a reviewer objects.** A four-way conjunction is the weakest possible shape
for a novelty claim — it is true iff no one has done all four, which is nearly
always true and nearly never interesting. Per the sweep, **provider-independent
enforcement** is occupied outright: EBTE (`arXiv:2607.25364`, "Server-Verified
Action Claims Without Trusting Model Rationales"), and by [18] and [19], which
the paper itself names two sentences later. **Evidence-bound domains** are partly
occupied by CHyD (`arXiv:2609.10046`), at decode time on span granularity. The
paragraph then concedes the loop to Guardrails/NeMo in its own final sentence, so
it argues against itself inside one paragraph.

**What to do.** Adopt the sweep's replacement sentence more or less verbatim:
lead with the two genuinely unoccupied conjuncts (numeric claim-value checking
against a deterministic producer; completeness against a ranked anchor), keep
"over a decidable faithfulness class", and demote provider-independence from a
contribution to a **design requirement the setting imposes**. Add the
curation-time (EG-VAR) / decode-time (CHyD) / **call-time** (yours) trichotomy —
it is real, crisp, defensible, and currently unstated.

### F12. Missing prior art an SE reviewer will personally know

Four gaps, in descending danger for this venue:

- **LeakageDetector 2.0** (`arXiv:2509.15971`, cs.SE, VS Code extension) — direct
  prior art for `detect_leakage`, and its LLM-driven guidance path is *an
  unguarded narrator over leakage evidence*, i.e. the exact surface your contract
  governs. Absent from the manuscript. At an SE venue this is the one that draws
  blood, and the honest framing is strong for you: they ship the detector plus an
  ungoverned explanation layer; you govern that layer.
- **Data-to-text / table-to-text faithfulness ancestry** — the manuscript engages
  none of it. Your entity channel is the hallucinated-entity ratio and your
  completeness channel is record coverage, both measured since 2017–2021
  (`liu2021entitycentric`, `dhingra2019parent`, `wiseman2017challenges`). Left
  out, this is the paragraph an NLP-adjacent reviewer rejects on. Written in, it
  is an asset: those works *measure and reduce*; an evidence-closed setting lets
  the same two quantities be *enforced*.
- **CHyD** (`arXiv:2609.10046`) — the only other 2026 work claiming a hard
  faithfulness guarantee. Cuts both ways and you should say so: it weakens "hard
  guarantee is unprecedented", and it *strengthens* your decode-time-unavailable
  argument, because CHyD is exactly what cannot run behind a commercial API.
- **Narration Gap** (`arXiv:2606.19588`) — state explicitly that its Theorem 3.2
  (monitor/enforcer equivalence) does not overlap Prop. 1 or Prop. 2, or a
  reviewer will assume it does. Also disambiguate its "complete monitor" from
  your "completeness channel"; same word, different predicate.

Also: **HALO** (`arXiv:2607.17883`) now owns "by construction" in this space, and
the manuscript uses the phrase three times. Cite it once and let the measurements
be the differentiator.

### F13. The value channel was silent because it is nearly unreachable by construction, and the paper gives the wrong reason

**Target.** §V-C: *"The honest explanation: the numbers sit verbatim in the
context window, copying them is easy, and the anchor is the single most salient
item in E."* And §III-C: *"each claim `{column, statistic, value}` must match the
measured value within 0.005"*.

**Why a reviewer objects.** The schema tells a different story.
`claim_schema["statistic"]` is `{"type": "string", "enum": ["correlation"]}` — a
one-element enum. The verifier then never reads `statistic` at all; it looks the
column up in `evidence_correlation_map` and compares. So the advertised
`{column, statistic, value}` triple is a pair, and the value channel covers
exactly **one** of the four quantity families §III-C Layer 1 says E carries
(the suspicious metric; per-feature correlations; the ranked candidate set;
P(ŷ = y)). The metric value and the perfect-match rate are not expressible as
claims, therefore not verifiable, therefore can only be misquoted in
`narration` — the unverified channel of F2.

The real explanation for 0/200 on value is not "copying is easy". It is that the
verified surface is one statistic, pinned to a column the enum already
constrains, so the narrator has almost nothing to get wrong inside the channel
and everything to get wrong outside it. That is a much sharper finding, it is
true, and a reviewer who opens the schema gets there in five minutes.

**What to do.** Describe the claim channel accurately — *"correlation claims, the
one statistic the current schema admits"* — and move the generalisation
(multi-statistic claims over metric, perfect-match rate, derived quantities) into
future work where it already half-lives. Then re-write §V-C's explanation to the
structural one. This finding also pays: it is the concrete reason the value
channel needs *widening*, which is a better research agenda than the current
"longer contexts could activate it".

### F14. A reviewer running artifact evaluation reaches a bad conclusion

Consolidating F3, F5, F6, F8 into what actually happens in an AE track:

| Check | Result |
| --- | --- |
| Clone and locate raw measurement data | Absent (F5) |
| Resolve the harness commit pinned in the run record | Dangling (F5) |
| Re-score published cells with the paper's own independent scorer | Impossible, no input (F5) |
| Run `evaluate --llm` and compare the panel to §I's description | Contradicted, pre-`96f0071` (F3) |
| Read recent commit titles | *"Fix four wrong statistics…"*, *"…an overclaimed README"* (F6) |
| Check "the guarded path is the only path" against `mcp_server.py` | Contradicted (F8) |

Any two of those produce a "Available" badge and no "Reusable"/"Functional"
badge. All six produce a reviewer who stops extending good faith. Fix the
`.gitignore`, commit the records, re-run one provenance cell, and disclose the
statistics fixes in the paper. This is a week of work that changes the review.

---

## Tier 3 — findings that will be raised, cheap to fix

### F15. The paper leans on numbers its own Table II proves are meaningless

Abstract and contribution 2 lead with *"Bare narrators fabricate at 11.5% and
43.5%"*. Table II then shows the same model on the same evidence spanning
1%–100% under rule-free rewording. A sharp reviewer will note that the paper uses
a point estimate as a property of the narrator in its abstract and then spends
§IV-C and §V-A arguing no such property exists. The fix is framing, not data:
present 11.5%/43.5% as *"two points inside a wording-dependent range"* from first
mention, and let Table II be the headline number it deserves to be. (It is, by
some distance, the best result in this paper. See the next section.)

Related, and worth pre-empting: `ablation_live_deepseek_2026-06-12_battery.md`
records that the same L1 synthetic configuration read **15.0%** on 2026-06-11 and
11.5% on 2026-06-12. The manuscript reports only 11.5% — the conservative
choice, to your credit — but the plan's discipline is *"every executed cell is
reported"*. One clause fixes it and buys credibility.

### F16. Internal number inconsistency: 76 vs 80

Abstract: *"under stress the verifier caught **76** live violations"*.
Contribution 2: *"the verifier caught **80** live violations"*. §VI: *"caught
**80** violations across the stress configurations, **76** of them on real data"*.
§IV-C reconciles them (76 real-data + 4 synthetic = 80), so the abstract is
simply wrong. Reviewers notice arithmetic in abstracts.

### F17. Prop. 1 is not a proposition

*"Deciding faithfulness of a structured response r … takes O(|r|) expected time
after O(m) preprocessing."* This is "hash tables are O(1)", dressed with a ∎.
Presenting it as one of two formal facts carrying the design invites exactly the
ridicule that F1 will already have earned. It also quietly elides worst case:
the abstract says *"verification is linear-time"*, the proposition says
*expected* time. Either state it as a remark in running text (my preference — it
loses nothing and costs a reviewer's goodwill nothing), or state it properly with
the worst case named.

### F18. Prop. 2's impossibility half rests on an unstated premise

*"Completeness is not safe-repairable: it is an existential requirement, and any
repair must add a reference the model did not produce."* A theory reviewer will
immediately propose the deletion-only repair the paper does not consider:
**delete the whole narration and render nothing**. That is pure deletion, it
attributes nothing, and it repairs the violation. The paper forecloses it only
by asserting *"the schema requires a verdict field"* — which makes the
impossibility a consequence of your own schema, not a structural fact about
narration.

Note also that your completeness constraint is *conditional* ("a verdict other
than `cannot_determine` must address the top-ranked candidate"), not purely
existential, which is why the escape exists at all. Be careful here when adding
the Chomicki citation per F1: the mapping to inclusion dependencies is looser
than a one-line citation implies, and a database reviewer will engage on the
details rather than accept the analogy wholesale.

The fix strengthens the paper. Make the premise explicit and make it *normative*:
the admissible output set is fixed by the product requirement that an
investigation returns a verdict; given that, suppression is not available; given
that, completeness admits no safe repair. Stated that way the proposition becomes
what it actually is — an argument about what repairs a *speaker* may perform —
which is the same territory as F1's salvaged contribution. The two fixes are the
same fix.

### F19. Threat model excludes prompt injection while the enum is built from third-party column names

§III-A rules out *"adversarial column names planted in user data as prompt
injection"*. But the evidence producer reads user CSVs, and the paper's own
real-data task *"starts from the public Insurance Charges dataset"*. Column names
from a third-party file flow straight into the Tier A enum. Ruling injection out
of scope is legitimate for a first paper; leaving unremarked that the binding
makes attacker-chosen strings into *schema-sanctioned, "verified" entities* is
not. Post-`96f0071` the renderer labels them `verified against the evidence`.
Verified ≠ safe, and one sentence saying so is cheap insurance.

### F20. Citation errors

- §I: *"experiment trackers log training runs [2], [3]"* and §II: *"Experiment
  trackers [2], [3] record metrics without interpreting them."* Reference [3] is
  Abadi *et al.*, "TensorFlow: A system for large-scale machine learning", OSDI
  2016. TensorFlow is a training framework, not an experiment tracker. You
  presumably want TensorBoard or W&B. An ML-systems reviewer catches this
  instantly and it is the kind of error that colours everything after it.
- [7] is described as *"tool-integration standards"*. MCP is a vendor protocol,
  not a standard. Soften to "a tool-integration protocol".
- [1], [15], [16] are bare URLs with no access dates; IEEE style wants them.
- Per the sweep, if `[guardagent]` or `[bfcl]` make it into the final reference
  list, both carry unresolved metadata (title drift; unverified volume/pages).

### F21. Figure 1 is a placeholder

*"**[INSERT FIGURE 1 HERE]**"*. The repo has `paper/contract_boundary.png` and
`paper/contract_flow.png`. Obviously a build-pipeline artifact of the Markdown
source, but do not let it reach a submission system.

### F22. The 37.5% → 1.3% comparison is presented without the machinery it needs

*"a ~28× drop in point estimate and at least ~5× at the bound"* compares a point
estimate against the *other* rate's interval bound. The 37.5% carries no interval
of its own. Separately, *"The model is rejected"* (the iid retry model, 114
predicted vs 76 observed) is asserted with no test statistic. Both are almost
certainly fine — the effect is enormous — but stating an effect that size without
an interval on both arms and without a test is exactly the sort of thing a
statistics-literate reviewer flags, and it is a five-minute fix. See also F10(d):
the causal claim attached to this number is the bigger problem.

### F23. L2 and L3 are indistinguishable on every measured channel, and the paper does not say so

Table I: L2 and L3 are both 0.0% (0/200) in both tasks, on all three channels.
The shipped L3 *includes* the strict prompt (§III-C Layer 2 sits inside the
contract). So on the paper's own data, the contract is empirically
indistinguishable from a good prompt; the separating evidence is entirely the
STRESS arm (prompt removed) and the structural argument. Your Round-2 plan
anticipates this exactly (LC-1, and §2.1's note that under the strict prompt
*"every arm would tie at zero and the block would be uninformative by
construction"*). Say it in the paper, in §V-A, before a reviewer says it first
and less kindly. It costs nothing — the structural argument is your actual claim
— and saying it demonstrates you understand your own design.

### F24. Terminology collision on "data leakage"

If `arXiv:2606.17114` or any privacy-side work enters the reference list,
disambiguate on first use: there "data leakage" is privacy exfiltration, here it
is train/test contamination. Also note the sweep's §3.1 correction — that paper
**cannot** be cited as field evidence of narrators fabricating tool evidence;
`arXiv:2608.11274`'s false-completion audit is the correct substitute and is
better evidence anyway.

---

## What this paper does better than most submissions — do not edit these away

Revision pressure destroys good things first. These are the parts that make the
paper worth revising rather than abandoning, and several of them are rarer than
the authors seem to realise.

1. **Table II is the best thing in the paper, and it may be the best thing in the
   area.** Six pre-written, rule-free paraphrases of one instruction, same model,
   same evidence, same schema, 1% → 100%. *"All six paraphrases were written
   before any measurement and none was discarded afterwards."* That sentence plus
   that table is a cleaner demonstration that prompt-level faithfulness cannot be
   certified from observed compliance than anything I am aware of in print. Every
   restructuring proposal below is built around promoting it. **Do not let it get
   compressed to make room for related work.**

2. **A pre-stated prediction was rejected and reported as rejected.** *"Before
   running we wrote down an iid retry model… it predicts about 114 catches. We
   measured 76. The model is rejected, and in the direction that matters."*
   Papers almost never do this. Keep it, keep the arithmetic, and keep the
   framing that the rejection is informative rather than embarrassing.

3. **Negative results are reported rather than buried.** *"The other two channels
   stayed silent everywhere… We do not read the silence as a victory."* The label
   in §V-C needs fixing (F10c) but the instinct is right and rare.

4. **Novelty is conceded where it should be.** *"We do not claim that varying a
   constraint with the input is new; [14] already does it."* And Limitation 5:
   *"Tier A steers; it does not enforce… without decode-time enforcement it stays
   exactly as uncertifiable as a prompt."* Conceding that your own most
   impressive-looking result (11.5% → 0/200 from the enum alone) is uncertifiable
   steering is the single most credibility-generating move in the manuscript.
   Protect it.

5. **The stress-arm confound is disclosed with numbers.** Limitation 3 gives the
   propensity mismatch (2% vs 11.5%, 37.5% vs 43.5%) and correctly refuses to
   read catches as a bare-rate estimate. Textbook.

6. **The scoping sentence in §I is exactly right.** *"every entity and every
   number in the validated citation and claim channels is present in, and equal
   to, the deterministic evidence"* — channel-scoped, provider-independent,
   falsifiable. The fix for F2 is to make §III-C match *this* sentence, not to
   weaken this one.

7. **The framing move — "this setting is easier than hallucination in general"**
   (*"Our starting point is that this setting is easier"*). Opening by narrowing
   your own problem is unusual and it is what makes the decidability argument
   land. Keep it in the abstract.

8. **Decidability as the licence for simple checks.** *"None of the checks is
   sophisticated, and that is the point: decidability makes simple checks
   sufficient."* Reviewers who see through complexity theatre will like this a
   great deal.

9. **The motivating sentence is the right one.** *"A wrong diagnosis delivered
   fluently is worse than no diagnosis, because the practitioner acts on it."*
   Keep it — and note it now also motivates the F6/F7 disclosure, since the
   deterministic layer produced exactly such diagnoses.

10. **Spearman alongside Pearson, with the reason given.** *"a log-target leak
    sitting at ρ_P ≈ 0.94 would slide under a Pearson-only filter."* Small,
    concrete, evidently from practice. This is what distinguishes a shipped tool
    from a paper prototype, and it is worth more than the word "production" was.

11. **The transposition section is disciplined.** §V-D names two target domains
    and immediately says *"We have not evaluated either domain; they mark where
    our claims stop."* Most papers claim generality here. This one marks a
    boundary.

12. **The pre-registration exists at all,** and `analysis_plan_2026-09.md` §1.5's
    refusal to hide behind cost (*"Cost is not the obstacle, and we will not
    pretend it is"*, with a table and a pre-committed N = 3,838 concession) is
    better methodological hygiene than most SE empirical work. Cite the plan in
    the paper. It is currently invisible to a reviewer, and it is an asset.

---

## Restructuring proposal for the contribution list

The current list is three items, and the first is half-borrowed (F1), the second
conflates two claim types (F10a), and the third — the best result — is last.
Restructure around the claim architecture the analysis plan already fixes.

### The problem to solve

Prop. 2 cannot carry contribution weight any more. Something has to replace it,
and the paper already contains three candidates that are stronger: Table II, the
decidable-class framing, and the safety argument that survives from Prop. 2.
The restructure is therefore a promotion, not a demotion — but it changes what
the paper is *about*, and the abstract has to change with it.

### Proposed contribution list

**C1 — The evidence-closed class, and why decidability changes the goal.**
*A narration task is evidence-closed if the set of citable entities and
verifiable quantities is finite and machine-enumerable at the moment of the
narration call.* For this class faithfulness is decidable, which moves the
sensible objective from mitigation to prevention on the decidable channels. This
is the framing contribution and it is genuinely the paper's own — nothing in the
sweep competes with "over a decidable faithfulness class". Xu's open-world
inevitability result (`arXiv:2510.05116`) is the theoretical complement and
should be cited here, in the *first* related-work paragraph: it argues
hallucination is inevitable under an open world and admits the closed-world case
may be mitigable; this paper supplies the constructive closed-world instance.

**C2 — The contract: call-time evidence binding plus provider-independent
re-verification, with the repair semantics a speaker requires.** Two sub-parts,
and the second is where Prop. 2's survivor goes:

  - *Mechanism.* Tier A binds the tool-schema enums to E **at call time** — the
    distinction to draw explicitly is against EG-VAR's **curation-time** lifts and
    CHyD's **decode-time** span constraint. Tier B re-verifies in code the
    provider does not control. Provider-independence is presented as a
    **requirement the setting imposes** (commercial APIs expose no token
    distribution), not as a novelty conjunct (F11).
  - *Repair semantics.* The universal/existential repair asymmetry is classical
    [chomicki2005minimal]; what narration adds is that insertion is not merely a
    different repair but an **inadmissible** one, because it attributes content to
    a speaker who did not produce it — and, in the corollary that appears to be
    unstated anywhere, substituting an abstention is inadmissible for the same
    reason. Hence strip-for-soundness, flag-for-completeness is *forced by the
    presence of a speaker*, not chosen. Present this as a **transposition with a
    safety argument**, explicitly, in those words. Make the normative premise
    visible per F18 (the product requires a verdict, so suppression is not an
    available deletion). Prop. 1 becomes a remark in running text (F17).

**C3 — The paraphrase sweep: prompt-level faithfulness cannot be certified from
observed compliance.** Promote this to its own numbered contribution and,
frankly, consider promoting it to the title's second clause. One model, one
evidence dictionary, six pre-written rule-free paraphrases, 1% → 100%. This is
the result that survives every objection in this review — F1 does not touch it,
F5 does not touch it (it is a *rate* claim, C-M, and its weakness is
non-transportability, which the paper already states), F9 does not touch it
because it needs no baseline, and it is what makes the structural argument
necessary rather than merely available.

**C4 — Live measurement of a shipped contract, with the claim architecture made
explicit.** Reported under the plan's split, in the paper, visibly:

  - **C-M (measured, per-cell, non-transportable):** L1 rates, the sweep, the
    first-attempt violation rate under stress, catches, the repeat-violation
    rate, and — newly required — abstention rates and the two-column omission
    reporting.
  - **C-S (mechanism, not established by any sample):** the entity and value
    invariants, supported by Prop. 1/2, the verifier source, the 23 (now 27)
    contract tests, the independent scorer, and the registered property-based and
    mutation testing.
  - **The concession, stated in the paper and not only in the plan:** the
    contract-arm omission zero *is* an ordinary empirical rate with a 1.88% upper
    bound, and 0% and 1.88% are not distinguishable at N = 200. Say it in the
    results section, in those words. Conceding it where it bites removes the
    reviewer's best line of attack on the other two channels, because it
    demonstrates you know which is which.

### Abstract and title consequences

- Drop *"Two short formal observations carry the design"* — after F1 and F17,
  neither carries anything and the sentence invites the exact scrutiny that hurts
  most. Replace with the decidability framing and the sweep.
- Drop *"production narrator"* (F6). Replace with "the shipped code path".
- Fix *"all three channels"* (F10a) and *"prevented outright"* (F10b): prevention
  is a two-channel claim; completeness is detect-and-flag, by your own
  proposition.
- Fix 76 vs 80 (F16).
- Promote the sweep into the abstract's second sentence rather than its seventh.

### Related work: the paragraph that must change most

Rebuild "Constrained and structured generation" around a **three-axis table** —
*when the constraint is bound* (curation / decode / call time), *who enforces*
(provider / own code), *what predicate is checked* (grammar, span-verbatim,
NLI-support, numeric equality, coverage) — with CHyD, EG-VAR, FAX,
ProvenanceGuard, EBTE, Guardrails, NeMo and this work as rows. A table makes the
conjunction claim unnecessary: the reader sees the empty cells. It also gives
you somewhere to put the six works the manuscript currently omits (F12) without
inflating the prose. And add the data-to-text ancestry paragraph — the one-liner
the sweep drafted (*"The entity and coverage channels are not new… What changes
in an evidence-closed setting is that both become decidable, so they can be
enforced rather than scored"*) turns your weakest omission into a positioning
sentence.

---

## The order I would fix these in

1. **F2, F3** — false statements about the artifact. Nothing else matters until
   these are gone. Both are text-only fixes plus one renderer test.
2. **F1** — the Prop. 2 recast. One paragraph, one citation, and contribution 1
   reworded. Converts the largest liability into a defensible framing move.
3. **F5** — fix `.gitignore`, commit what records exist, say plainly what does
   not, re-run one cell for provenance.
4. **F6, F7** — drop "production", disclose the four statistics fixes, add the
   evidence-consistency-not-truth sentence and the laundering observation.
5. **F10, F8, F11, F12** — align with the plan, correct the surface claim, narrow
   the novelty claim, add the six missing works.
6. **F13, F23, F4** — the three "say the true thing about your own result"
   fixes. Each costs one or two sentences and each removes a reviewer's best
   line.
7. Tier 3 in a single editing pass.

Items 1–4 are, by my estimate, about a week of work and they are the difference
between a paper that gets read and one that gets used as an example.
