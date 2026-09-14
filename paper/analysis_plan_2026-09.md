# Pre-registered analysis plan — Round 2 (baselines, providers, silent channels)

- **Version:** 2.0 — 2026-09-14 (written BEFORE any Round-2 live call; must be
  committed before the first one)
- **Scope:** the three measurement additions for the journal version —
  enforcement-mechanism baselines (roadmap R3/R4), the multi-provider extension
  (R1/R5), and the deliberate pressure battery on the value and omission
  channels (R2). It also fixes the claim architecture (§1) that the INISTA
  manuscript currently leaves implicit.
- **Relationship to `paper/analysis_plan.md` v1.0 (2026-07-07):** v1.0 stays in
  force unamended, and nothing in this file revises it. Where this file is more
  specific (arm definitions, contrast precedence, licensing rules) it narrows
  v1.0 inside v1.0's own allowances. Where a rule here and a rule there could be
  read as conflicting, **v1.0 governs** and the conflict is logged in §9.
  Hypothesis numbering continues from v1.0's H1-H6; Round 2 adds H7-H16.
  References of the form "A3.x" point into v1.0 §8.
- **Discipline (inherited, unchanged):** every executed cell is reported;
  deviations are logged in §9 with a reason BEFORE the deviating run; a
  mock-mode number is never discussed as a measurement; no rate appears without
  raw k/N and its interval; predictions are never revised after seeing data.

---

## 0. Round 2 in one paragraph

The published battery (June 2026, `deepseek-chat`, two tasks) established a bare
narrator rate of 11.5% and 43.5%, a paraphrase span of 1% to 100%, a Tier B
first-attempt catch rate of 37.5% under stress with repeat violations at 1.3%,
and 0/200 on every channel of every contract cell. Three objections survive that
battery: nothing was compared against an alternative enforcement mechanism, one
model carried every number, and two of the three verified channels never fired.
Round 2 addresses exactly those three, and fixes in advance what each possible
outcome is allowed to support. Round 2 adds no new claim type: it adds arms,
providers, and evidence pressure to claim types already registered in v1.0.

---

## 1. Claim architecture: what rests on measurement, what rests on mechanism

This section exists because a reviewer will read "0/200, Wilson 95% [0.00,
1.88]" and say, correctly, that 0% and 1.8% are indistinguishable at that N.
The objection is right about one claim and beside the point on another, and the
manuscript does not currently separate them. This plan fixes the split before
Round 2 data exists, so it cannot be arranged afterwards to fit a result.

### 1.1 The measurement claim (C-M)

**C-M is about the narrator, not the contract.** It is the load the contract has
to absorb, and it is estimated from samples:

- bare (L1) per-channel rates, per provider, per task, per prompt variant;
- the paraphrase spread (Table II), which is the sharpest of these numbers;
- Tier B first-attempt violation rate under stress (q), total catches, the
  retry-recovery rate, the residual strip rate, the repeat-violation rate;
- abstention rate, claims per response, latency, calls, token cost;
- **the contract-arm omission rate** (see §1.3 — this one belongs here, not in
  C-S, and that is the part the manuscript currently blurs).

Every C-M quantity gets raw k/N and a Wilson 95% interval, is scoped to its
(provider, model pin, task, evidence hash, prompt variant, arm, date) cell, and
is not transportable. The paraphrase sweep is this project's own proof of
non-transportability: one model, one evidence dictionary, six innocuous
rewordings, 1% to 100%.

### 1.2 The mechanism claim (C-S)

**C-S is about the code, and no sample establishes it.** On the entity and
value channels a user-facing violation is not rare, it is absent by
construction: Tier B tests set membership and a float difference and deletes
whatever fails, before anything is rendered (Prop. 1, Prop. 2, and A3.1, which
already labelled these cells analytic and withdrew the L1-vs-L3 Fisher test as
vacuous). The support for C-S is:

1. Prop. 1 (linear-time decidability) and Prop. 2 (safe repair by deletion);
2. the verifier source in `src/mlcompass/agents/leakage_investigator.py`;
3. the 23 shipped end-to-end contract tests against mock clients;
4. `scripts/independent_scorer.py`, which recomputes all three channels from raw
   JSONL without importing the product's helpers (A3.9a);
5. the property-based and mutation testing registered in §1.6.

None of these is a rate, and none of them improves with N.

### 1.3 Where the reviewer's objection is conceded in full

The channels are not symmetric, and the concession has to be made precisely
where it bites:

- **Entity and value channels, contract arms.** A 0/200 cell is an
  implementation-consistency check. It reports that in 200 live responses the
  implementation never contradicted the invariant. It is *not* an estimate of a
  residual violation rate, and its Wilson upper bound must not be read as a risk
  bound. The reviewer's objection does not apply because no rate claim is made.
- **Omission channel, contract arms.** Completeness is detect-and-flag, not
  strip (Prop. 2). A contract-arm omission number is therefore an ordinary
  empirical rate carrying all the weakness a 0/200 has, **and the manuscript
  concedes the point in those words.** 0/200 with an upper bound of 1.88% is a
  weak statement about completeness, and this plan says so before the data.
- **Undetected omission** (a committed verdict that skips the anchor and is
  never flagged) is a different object: it would be a verifier defect, handled
  under §3.5, not a rate.

Reporting consequence, frozen now: every contract-arm results table carries the
omission channel in **two columns** (flagged / undetected), and every analytic
cell is labelled analytic in the table itself, not in a footnote.

### 1.4 The reading rule for every 0/N cell

Frozen wording pattern for the manuscript, to stop drift between drafts:

> 0/200 (Wilson 95% [0.00, 1.88]). On the entity and value channels this is a
> consistency check on the implementation of an invariant that holds by
> construction (Prop. 1, Prop. 2), not an estimate of a violation rate; the
> rate the contract absorbs is measured separately in the unguarded arms of the
> same cell (first-attempt violations: k/N).

### 1.5 What N an empirical demonstration would require

If a reviewer insists the guarantee be shown empirically rather than argued
structurally, the arithmetic is fixed here so it cannot be negotiated later. For
k = 0 the Wilson 95% upper bound is z²/(N + z²), z = 1.96:

| Target upper bound on a 0/N cell | N required | Wilson upper achieved |
| --- | ---: | ---: |
| 5% | 73 | 4.99% |
| 2% | 189 | 2.00% |
| 1% | 381 | 1.00% |
| 0.5% | 765 | 0.50% |
| 0.1% | 3,838 | 0.100% |
| 0.01% | 38,411 | 0.0100% |
| 0.001% | 384,143 | 0.00100% |
| 0.0001% (1 in a million) | 3,841,455 | 0.000100% |

These are **per cell**. The Round-2 contract-arm footprint alone is on the order
of 40 cells (providers x tasks x arms), so a per-cell bound of 0.01% is roughly
1.6M live responses and a per-cell bound of 0.0001% is roughly 161M.

Three observations, all to be stated in the manuscript rather than used as an
excuse:

1. **Cost is not the obstacle, and we will not pretend it is.** At the pinned
   DeepSeek price (about $0.00027 per call at the battery's measured token
   profile), a 0.01% bound across roughly 42 cells is about $450. We can afford
   the reviewer's demand at that resolution. At 0.0001% per cell it is about
   $45,000, which is prohibitive, but the honest headline is that the first
   rows of the demand are affordable.
2. **Calendar and configuration stability are obstacles.** At a sustained 5
   calls per second, 0.0001% per cell is about 224 hours per cell and over a
   year for the full set. The June battery's `deepseek-chat` alias was
   deprecated within about six weeks of the run. The configuration such a
   sample would certify stops existing before the sample finishes.
3. **The sample does not buy what is being asked for.** Any such sample is
   conditional on one model version, one wording, and one evidence shape, and
   this project has already measured that the first two axes move the rate by
   two orders of magnitude. An empirical bound on a contract cell certifies
   that cell; the structural argument certifies the class.

**Pre-committed concession (fixed now so it cannot be chosen after seeing
data):** if the demand is made in review, we run **N = 3,838 per contract cell**
(0.1% upper bound, ten times the current resolution) on the cheapest pinned
provider for the two P0 tasks, and report it labelled as a consistency check at
higher resolution, not as a certification. No other N is negotiated.

### 1.6 The pre-registered falsifier for C-S

C-S is falsifiable, cheaply, and not by sampling the narrator. Registered now as
the empirical complement that scales:

- **Property-based search.** A generator over (evidence dictionary, candidate
  response) pairs, at least one million generated cases, asserting that the
  Tier B output contains no entity outside the evidence set and no claim
  outside tolerance.
- **Mutation testing of the verifier.** Every mutant of the Tier B checks must
  be killed by the shipped test suite, or the survivor is documented in the
  manuscript's threats section.
- **Falsifier:** a single generated pair for which Tier B returns a response
  carrying an off-evidence entity or an out-of-tolerance claim refutes C-S
  outright. That outcome is reported as a refutation, the invariant claim is
  withdrawn or narrowed, and the manuscript changes accordingly.
- Zero API calls. Building it belongs to `contract-test-engineer` and
  `ci-engineer`; it is registered here because it is the evidence C-S actually
  rests on.

### 1.7 Manuscript wording rules (frozen)

**Banned:** "the contract achieved a 0% fabrication rate"; "reduced fabrication
to zero" (both read as estimates); "100% reliable"; "fabrication is impossible"
without naming the channel; any rate without k/N and an interval; "across
providers" or "in general" attached to any measured rate; a mock number
described as a measurement.

**Required:** the §1.4 pattern on every contract-arm zero; the provider and
model pin, task, evidence hash, prompt variant and date on every unguarded rate;
the two-column omission reporting of §1.3; the analytic-or-empirical label in
every results table.

---

## 2. Addition 1 — baseline comparison (enforcement mechanisms)

### 2.1 Arms

All arms below run the **same task, same evidence dictionary (identical evidence
hash), same model pin, same user message, same sampling pin, same repair budget
(2), same scorer**. They differ only in the enforcement mechanism.

| Arm id | Mechanism | What it enforces | Status |
| --- | --- | --- | --- |
| `A-L1` | bare prompt, open schema, no verification | nothing | exists (`layer1`) |
| `A-CONTRACT` | bare prompt + Tier A call-time enum + Tier B | entity and value by construction; omission flagged | **new arm `layer3_bare`**, harness work |
| `A-STRESS` | bare prompt + Tier B only (no enum) | same, with first-attempt violations observable | exists (`layer3_stress`) |
| `A-GR-STOCK` | Guardrails AI stock loop: JSON/type validation + reask, no faithfulness validator | output shape | adapter to be written |
| `A-GR-OURS` | Guardrails AI loop carrying our Tier B checks as a custom validator | loop parity probe | adapter to be written |
| `A-NEMO` | NeMo Guardrails, nearest output-rail configuration with an equivalent validator | toolkit 2 | adapter + config to be written |
| `A-STRICT-STATIC` | provider strict mode, static schema (types + required, **no** call-time enum) | shape, decode-enforced | exists (`--strict` on open arms) |
| `A-STRICT-ENUM` | provider strict mode + call-time evidence enum, **no** Tier B | entity, decode-enforced | exists (`tier_a` + `--strict`) |

Three things are fixed here because they decide whether the comparison means
anything:

- **`A-CONTRACT` is the contract arm for this comparison, not the shipped L3.**
  Shipped L3 carries the strict prompt, so comparing it against a bare-prompt
  baseline would confound prompt with mechanism. `A-CONTRACT` is the shipped
  enforcement stack with the faithfulness prompt removed.
- **The comparison runs under the bare prompt.** Under the strict prompt this
  model already reads 0/200 at L2, so every arm would tie at zero and the block
  would be uninformative by construction. That reason is recorded now, before
  the run. A deployment-realistic strict-prompt replicate of the same arms runs
  at N = 200 on P0 only, is *expected* to be uninformative, and is reported
  regardless (no selective reporting).
- **`A-GR-OURS`'s validator is ours.** The manuscript says so in the arm's own
  caption. That contrast compares loops, not validators.

### 2.2 Contrast precedence (frozen)

- **Primary confirmatory contrast (X1): `A-CONTRACT` vs `A-GR-STOCK`**, composite
  user-facing violation rate (entity OR value OR flagged omission), real-data
  task, per provider. This is the deployed-default question: what a practitioner
  gets from the leading validate-and-reask toolkit out of the box versus what
  the contract gives, on identical evidence.
- **Primary differentiating contrast (X2): `A-CONTRACT` vs `A-STRICT-ENUM` on a
  non-entity channel**, conditional on §4 activating one. This is the only
  contrast that can separate the contract from decode-enforced constrained
  output on measurement rather than on capability. If §4 produces a persistent
  zero, **X2 is void** and the differentiation reverts to the capability
  argument (roadmap R11(c)), which is documentation, not data. That dependency
  is written down here so nobody discovers it after the run.
- **Secondary (X3): `A-CONTRACT` vs `A-STRICT-STATIC`**, entity channel.
- **Secondary (X4): `A-CONTRACT` vs `A-GR-OURS`**, all channels plus calls and
  latency. Loop parity.
- **Secondary (X5): `A-CONTRACT` vs `A-NEMO`**, all channels.
- **Descriptive anchor:** `A-L1` in the same cell, so every baseline is read
  against the unguarded floor of that exact configuration.

Multiplicity: Holm within the family {X1, X3, X4, X5} per provider; X2 is a
single pre-specified test and is not corrected. Descriptive tables carry
intervals and no tests (v1.0 §5, unchanged).

### 2.3 Metrics and interval methods

- Per-channel and composite user-facing violation rates, raw k/N, **Wilson 95%**
  single-proportion intervals (unchanged from v1.0).
- Differences between arms: **Newcombe hybrid-score 95% intervals**. Wald
  intervals are prohibited here; zero cells are the normal case and Wald is
  degenerate on them.
- Hypothesis tests: **two-sided Fisher exact**, alpha = 0.05.
- Reported per arm, always: provider calls per response, wall-clock latency,
  token usage and cost, abstention rate, error and empty counts, and
  `rejection_kinds` composition where the arm has a verifier.

### 2.4 N, resolution, and what the design cannot see

N = 200 per cell (P0 providers), matching the published battery. The resulting
resolution, computed in advance:

- Against a 0/200 arm, **six events in the comparator arm** is the Fisher
  significance threshold (k = 5 gives p = 0.061; k = 6 gives p = 0.030). That
  threshold stays near six at any balanced N; what a larger N buys is the
  probability that a small true rate produces those six events.
- Power at N = 200 vs 200: 0.21 at a true comparator rate of 2%, 0.56 at 3%,
  **0.81 at 4%**. The MDE is therefore about 4%. At N = 400 the same power
  arrives at about 2%.
- **Equivalence band delta = 2 percentage points**, chosen because it is the
  smallest band this N can resolve (0/200 vs 0/200 gives a Newcombe interval of
  plus or minus 1.88pp). Consequence, stated before the run: **this design
  cannot distinguish two mechanisms that differ by less than about 2pp.** That
  is precisely why the mechanism claim (§1.2) is not made by this comparison.
- Superiority is claimed only when the Newcombe interval for the difference
  excludes 0 **and** the point difference is at least 5pp. A difference that is
  significant but smaller than 5pp is reported as significant and not material.

### 2.5 Hypotheses and their rejecting outcomes

- **H7 — a stock validate-and-reask loop leaves the faithfulness channel open.**
  `A-GR-STOCK` shows a composite violation rate above 0 with k >= 6 at N = 200
  on the real-data task, landing within delta of the matched `A-L1` cell (the
  loop validates shape, so it should neither help nor hurt faithfulness).
  **Rejected if:** `A-GR-STOCK` reads 0/200 on all three channels wherever the
  matched `A-L1` cell exceeds 3%, or its composite rate is more than 10pp below
  `A-L1` with the Newcombe interval excluding -10pp. Either rejection is a real
  finding about the toolkit and is reported as one.
- **H8 — a static strict schema constrains shape, not content.**
  `A-STRICT-STATIC` shows an entity rate above 0 where the matched `A-L1` cell
  exceeds 3%, within delta of `A-L1`.
  **Rejected if:** `A-STRICT-STATIC` reads 0/200 on entity while `A-L1` exceeds
  3% in the same cell. That outcome would mean the strict envelope itself steers
  the narrator without an enum, which weakens our framing. It goes in the
  abstract, not a footnote.
- **H9 — a decode-enforced call-time enum ties the contract on the entity
  channel. We predict the tie and pre-concede it.** `A-STRICT-ENUM` reads 0 on
  entity and its Newcombe difference from `A-CONTRACT` lies inside delta.
  **Rejected if:** `A-STRICT-ENUM` produces k >= 1 on entity, which would be a
  provider strict-mode enforcement failure. That rejection favours us and is
  therefore held to the higher bar of §3.5: every event is adjudicated against
  the raw payload before it is reported.
- **H10 — channel coverage, not entity rate, separates the contract from decode
  enforcement.** Conditional on §4 activating the value or omission channel: in
  that activated cell `A-STRICT-ENUM` shows k >= 6 on the activated channel
  while `A-CONTRACT` shows 0.
  **Rejected if:** the activated channel reads 0 in `A-STRICT-ENUM` too, or §4
  never activates a channel (X2 void). Then the coverage claim has no empirical
  support in this campaign and reverts to the documented capability table
  (LC-3).
- **H11 — loop parity: the checks do the work, not the architecture.**
  `A-GR-OURS` matches `A-CONTRACT` on all three channels inside delta and
  differs only in calls and latency.
  **Rejected if:** any channel difference falls outside delta with the Newcombe
  interval excluding 0. Then the difference is attributable to the loop, and the
  contribution claim changes from "the verifier" to "the verifier plus its
  placement". Both outcomes are publishable and the manuscript sentence changes
  accordingly.

### 2.6 Losing conditions, written down before the run

These are the outcomes under which the contract **fails to outperform** a
baseline. Each is a sentence we commit to publishing, in the results and in the
abstract where it bears on a headline.

- **LC-1 (channel parity).** If, on a channel, a baseline's user-facing rate has
  a Wilson upper bound no worse than the contract's **and** the Newcombe
  interval for the difference lies entirely inside plus or minus 2pp, we report:
  *"on this channel, at this N, the contract does not outperform `<baseline>`;
  any remaining distinction rests on coverage and certifiability, not on the
  measured rate."* Given H9, we expect to publish exactly that sentence about
  the entity channel versus decode-enforced enums.
- **LC-2 (whole-comparison underpowering).** If no baseline arm exceeds 2% on
  any channel in any cell, the baseline block is underpowered to separate
  mechanisms at N = 200 and is reported as an inconclusive comparison, not as
  superiority. We do not silently raise N afterwards to rescue it: any N change
  is an amendment in §9 logged before the re-run, and both the original and the
  extended result are reported.
- **LC-3 (coverage claim without data).** If §4 yields a persistent zero, X2 is
  void and the sentence "the contract covers channels decode enforcement cannot
  express" is downgraded from a measured result to a capability argument backed
  by the provider-by-keyword support table, with the downgrade stated in the
  results section.
- **LC-4 (cost).** If the contract's calls, latency, or token cost exceed a
  baseline's by more than 25% at channel outcomes inside delta, that is a cost
  the contract pays for nothing measurable in that cell, and it appears in the
  same table as the rates, not in a separate favourable framing.
- **LC-5 (the architecture loses to its own checks).** If H11 holds while
  `A-GR-STOCK` is bad and `A-GR-OURS` matches `A-CONTRACT`, the honest reading
  is that the verifier is the contribution and the loop is interchangeable. The
  manuscript then claims the verifier plus the call-time binding, and explicitly
  not the architecture.

### 2.7 Fairness controls

- Identical evidence hash across every arm of a comparison; the hash appears in
  every table.
- Identical repair budget: 2 reasks for the toolkits, 2 corrective retries for
  the contract. A toolkit that cannot be configured to budget 2 runs at its
  nearest setting and the discrepancy is stated in the arm caption.
- All arms of one comparison run within the same 24-hour window, with per-arm
  start and end timestamps recorded. If a provider incident splits a comparison
  across days, the cell is re-run in full rather than stitched.
- All arms are scored by `scripts/independent_scorer.py` from committed JSONL.
  No toolkit's self-reported pass or fail is used as an outcome; toolkit
  telemetry is recorded as a separate descriptive column.
- Adapter code (Guardrails, NeMo, the strict-mode paths) is committed and
  mock-tested before any live call, and the adapter commit hash is pinned in the
  run record beside the harness commit.
- **Adapter-caused failures are data, not exclusions.** If a toolkit's loop
  crashes, loops forever, or cannot express a check, that is a property of the
  toolkit: logged, counted, reported. Only transport failures are excluded
  (v1.0 §7). A toolkit that cannot express a check is scored as unenforced on
  that channel and marked "not expressible" in the capability table; it is never
  dropped from the comparison.

---

## 3. Addition 2 — multi-provider extension

### 3.1 Round-2 panel (frozen here; changes are amendments)

Minimum panel for Round 2, all pins from v1.0 §2 and the harness `PROVIDERS`
table:

| Role | Provider / model pin | Why it is in |
| --- | --- | --- |
| Continuity | `deepseek-v4-flash` (DeepSeek) | the published battery's successor pin |
| Strict-capable closed | `gpt-5.4-mini` (OpenAI) | the lane where opt-in strict mode is documented and switchable, so H8 and H9 exist at all |
| **Open-weights, self-hosted** | `Qwen/Qwen2.5-3B-Instruct` (local vLLM) | the only lane where decode-enforced constrained generation can be run |

Additions are permitted and must be logged in §9 before the run. **Removals are
not permitted once a cell has been executed**: a provider that produces
inconvenient numbers stays in the manuscript.

Two guards fixed now:

- The open-weights model is far smaller than the closed pins. Any difference in
  its rates is confounded with model size and tier, so **no open-versus-closed
  claim may be made at this panel size**, in either direction. The open-weights
  lane is there for provider generality and for the decode-enforcement
  comparison, not for a claim about openness.
- The asymmetry that decode-enforced generation is available only on a model you
  host is itself part of the argument (v1.0 §6) and is reported as such, not
  treated as a nuisance.

### 3.2 Per-provider or pooled: the decision

**Decision: all rate claims are per-provider. Pooling across providers is
prohibited.** The justification, fixed before the data:

1. **There is no estimand.** A pooled rate estimates the fabrication rate of a
   mixture whose weights are our arbitrary sampling design. No operator faces
   that mixture. The per-provider rate answers the question someone actually
   has.
2. **Heterogeneity is the finding.** H1 predicts at least a fivefold spread
   across providers under the zero-safe decision rule of A3.6. Pooling would
   average away the result the campaign exists to produce.
3. **This project's own sweep removes the ground for pooling.** Within one
   model, one evidence dictionary, six innocuous rewordings gave 1% to 100%. If
   the rate is not stable under paraphrase *inside* one provider, there is no
   provider-level parameter for a cross-provider pooled estimate to converge to.
   Pooling would assume exchangeability across providers immediately after we
   measured non-exchangeability across wordings of a single prompt one level
   down.
4. **The interval would be wrong.** A pooled Wilson interval assumes iid
   Bernoulli draws. Our draws are iid only within a
   (provider, model pin, task, evidence hash, prompt variant, arm) cell. Across
   providers they are stratified with unequal, unstable and unknown strata
   means, so the pooled interval understates variance. A random-effects summary
   would be the correct instrument, and three strata cannot identify a variance
   component.
5. **What we report instead.** Per-provider cells; the minimum and maximum with
   their intervals as the descriptive summary; the A3.6 zero-safe ratio rule for
   any spread claim. Meta-analytic pooling is off the table at Round 2 and would
   have to be pre-registered separately at the full seven-to-nine-family panel,
   before that data exists.

### 3.3 The one permitted pooling exception

A **conjunction bound** over contract arms may be reported as a supplementary
line: "0 events in M responses across the B enumerated cells below". It is
admissible there and nowhere else, because the claim it supports is universally
quantified (no cell produced an event), so the pooled figure is a count over the
cells actually run rather than an estimate of a population rate. It is never the
headline, the per-cell table always appears with it, and it is never computed
for L1, baseline, or sweep arms. This restates A3.12 and does not loosen it.

### 3.4 Hypotheses and their rejecting outcomes

- **H12 — wording effects do not transport across providers.** With the
  six-variant sweep replicated at N = 100 per variant on the two new providers,
  at least one variant moves by 20pp or more between providers, **or** the rank
  ordering of the six variants differs between providers (Kendall tau below 1).
  **Rejected if:** all six variants land within plus or minus 10pp across all
  three providers **and** the orderings are identical. That rejection would mean
  wording effects are a property of the prompt rather than of the model, which
  weakens the non-transportability argument this project leans on, and it goes
  in the discussion as a correction to our own framing. Pooling stays prohibited
  at Round 2 regardless, because §3.2 rests on four reasons and H12 is one.
- **H13 — per-provider contract invariance (an implementation check, not a
  rate).** On each provider the contract arms read 0 on entity and value.
  **Rejected if:** any contract arm yields k >= 1 on entity or value. Per §1.2
  that is not a worse-model finding, it is a defect finding, and §3.5 governs
  what happens next.
- Cross-provider inferential claims remain reserved for N = 200 cells; N = 100
  cells stay descriptive (A3.7, unchanged).

### 3.5 What happens if a verified channel is ever non-zero

Pre-committed procedure, so the response is not improvised while a headline is
at stake:

1. The run is not discarded and the number is not re-scored into silence.
2. The raw JSONL record, the evidence dump and the contract's post-strip output
   are attached to the run record.
3. The event is adjudicated against the independent scorer **and** by hand.
   Three outcomes, each reported: (a) scorer defect, (b) **verifier defect**,
   (c) genuine invariant violation.
4. Under (b) or (c), C-S is withdrawn or narrowed in the manuscript, the defect
   and its fix are described, and every prior cell is re-scored with the
   corrected verifier. A repaired verifier does not retroactively launder the
   earlier claim; the timeline appears in the paper.
5. The same procedure applies to the H9 rejection direction (a strict-mode enum
   escape), so a finding that flatters us is held to the same bar as one that
   does not.

---

## 4. Addition 3 — the two silent channels

### 4.1 Division of labour

`contract-test-engineer` designs the evidence that puts pressure on value and
omission, and owns the deterministic tests. This plan does not design that
evidence. It fixes, before any of it exists, (a) what an instance must satisfy
to be admissible, (b) what counts as activation, and (c) what each outcome
licenses as a claim.

### 4.2 Admissibility of a pressure instance (all five required)

An instance enters the battery only if, **before any live call**:

1. **The channel is mechanically reachable.** An offline check demonstrates that
   a scripted plausible-but-wrong answer is rejected by the shipped Tier B on
   that instance, and that a correct answer passes. No API calls; the same
   discipline as `fetch_fabbench_datasets.py --verify`.
2. **It is not a rounding trap.** `VALUE_TOLERANCE = 0.005` lets two-decimal
   rounding pass by design. An instance whose only pressure is rounding is
   inadmissible: a catch there would be an artifact of the tolerance, not a
   fabrication.
3. **The anchor stays well defined.** Omission pressure works by lowering anchor
   salience (crowding, near-ties, ordering, length), never by removing the
   anchor or making `candidate_leak_columns[0]` ambiguous. Anchor-free shapes
   stay excluded from omission denominators (A3.3).
4. **It is producible, not hand-written.** The evidence must come out of the
   shipped `detect_leakage` on a real or synthetic frame. Hand-authored evidence
   dictionaries are inadmissible: a reviewer will find the contrivance, and they
   would be right to.
5. **It is frozen in §9 with its spec, seed and offline verification output
   before the first live call on it.**

Instance #14 `synthetic_crowded` (A3.4) is already frozen in v1.0 and carries
into this battery unchanged.

### 4.3 The minimum battery a zero must survive

A zero is publishable as a boundary only if all of the following ran, each at
N = 200, each reported:

- **Pressure families, both:** (i) derived-value pressure, where the number the
  narrator must report is not present verbatim in the evidence; (ii) crowded or
  near-tie pressure, where correlations differ inside the third decimal
  (instance #14 and at least one real-data analogue).
- **Anchor-salience pressure** for the omission channel: in at least one
  instance the anchor is not first in any ordering the narrator sees.
- **Arms:** `A-L1` (unguarded), `A-STRESS` (first-attempt violations visible),
  and a contract arm, on every instance.
- **Worst-wording condition:** the `mechanical` paraphrase, which fabricates at
  100/100 unenforced on the entity channel, on at least one pressure instance.
- **Providers:** at least two, including the open-weights model.
- **Abstention below 80%** in every cell counted (A3.12). A cell at or above 80%
  abstention is *unevaluable* and licenses nothing; the zero there is trivial.

If any element is missing, the outcome is "not yet measured", not a boundary.

### 4.4 Activation criterion and the adjudication gate

- A channel is **ACTIVATED** in a cell when an unguarded arm shows **k >= 5 at
  N = 200** (Wilson 95% [1.07, 5.72]). The threshold sits above 1 on purpose: a
  single anomalous response must not carry a channel-activation claim.
- **Adjudication gate.** The first non-zero on a channel that has read zero in
  every published cell is likelier a scoring defect than a new model behaviour.
  Every event in the first activated cell is adjudicated by hand against the raw
  payload and the evidence dump before the activation is reported. The
  adjudication outcome is reported either way, including "the events were a
  scoring artifact and the channel remains silent".
- k between 1 and 4 is reported with k/N and its interval and licenses nothing
  beyond its own sentence (§4.5).

### 4.5 What a non-zero licenses

A ladder, fixed now:

| Observation | Licensed claim | Explicitly not licensed |
| --- | --- | --- |
| 1 <= k <= 4, unguarded, N = 200 | "we observed j value-channel events in 200 responses under pressure condition P on provider X (k/N, CI)" | "the channel fires"; any rate for the channel |
| k >= 5, unguarded, adjudicated | "under evidence configuration P on provider X the unguarded narrator produced value fabrications at k/N [LB, UB]; Tier B rejected all of them; y recovered on the named retry; z were stripped" | a general rate; a cross-provider rate; "this is a common failure mode" |
| k >= 5 on two or more providers including the open-weights one, same P | "the channel is not an artifact of one provider" | generalisation beyond configuration P |
| k >= 5 in two or more distinct pressure families | "the channel is reachable by more than one mechanism"; roadmap R2 closes and the two-thirds-unexercised objection retires | any claim about field prevalence |
| any event in a **contract** arm on entity or value | nothing: this is a defect, handled under §3.5 | reporting it as a rate |
| omission events in a contract arm | "detected and surfaced" (flagged), reported as a rate with its interval per §1.3 | "prevented" (Prop. 2 forbids the word) |

Whatever fires, the paired empirical quantities are reported with it:
first-attempt violation rate, catch count, named-retry recovery rate, residual
strip rate, abstention rate, and the post-strip completeness re-check outcome.

### 4.6 What a persistent zero licenses

A persistent zero under the full §4.3 battery is a **publishable boundary**, and
this plan says so before anyone has seen the data.

**Licensed:**

- "Across B pressure configurations and M total unguarded responses designed to
  elicit them, we observed no value fabrication and no critical omission (0/M,
  Wilson 95% [0.00, U]), with abstention below 80% in every cell and with each
  instance verified offline as one on which Tier B does reject a wrong answer."
- "In evidence-closed narration where the values sit verbatim in the context and
  the anchor is enumerable, the observed failure mode is entity fabrication.
  Value and completeness failures did not appear even under deliberate pressure
  at this N."
- "The two channels are therefore verified pre-emptively rather than in response
  to an observed failure. The price is a set lookup and a float comparison
  (Prop. 1), so the design decision does not depend on the rate."
- The scope statement that makes it a boundary rather than an absence:
  narration over **derived** quantities, longer evidence, multi-step chains and
  models outside this panel remain untested, and those are the conditions under
  which we expect the channels to activate.
- Roadmap R2 is downgraded per A3.3, which already pre-decided this so the
  definition of done cannot fail on a hope.

**Not licensed:**

- "The value and omission checks are unnecessary." The zero is conditional on
  exactly the axes the paraphrase sweep showed to be volatile.
- "The contract prevents value fabrication." Nothing was there to prevent.
- Treating the zero as support for C-S. C-S does not rest on it (§1.2), and
  using it that way would reintroduce the confusion §1 exists to remove.
- Any implication that the silent channels are safe in other narration tasks.

### 4.7 Hypotheses and their rejecting outcomes

- **H14 — the value channel activates under derived-value or near-tie
  pressure.** At least one admissible pressure cell reaches k >= 5 at N = 200 in
  an unguarded arm on at least one provider.
  **Rejected if:** every cell of the §4.3 battery reads k <= 4. Reported as the
  §4.6 boundary, not as a failed experiment.
- **H15 — the omission channel activates under low anchor salience.** Same
  criterion, omission channel, in a crowded or low-salience instance.
  **Rejected if:** every cell reads k <= 4 with abstention below 80%. Same
  boundary treatment. A cell at or above 80% abstention is unevaluable and
  neither supports nor rejects H15; the table must say so.
- **H16 — the post-strip completeness re-check is exercised live.** At least one
  observed response where stripping an unsound claim orphans the anchor and the
  re-check raises an omission flag a pre-strip check would have missed.
  **Rejected if:** no such case occurs across the whole battery. Reported as:
  the re-check path remains covered only by unit tests and has never been
  observed to fire on live data. That is a real limitation of the evidence for a
  code path the paper describes, and it goes in the threats section.

---

## 5. Cells, N, seeds, determinism

- **N:** 200 per cell everywhere in Round 2 except the sweep replication
  (N = 100 per variant, per v1.0 §3). Smoke runs (N = 5) are never data.
- **Evidence seeds:** `--seed 0` for every synthetic frame, matching the frozen
  instance specs (A3.4). The seed governs **evidence construction only**.
- **No provider-side seed is sent, ever.** Identical prompts under a fixed
  provider seed would collapse N draws toward one draw, and the cell would
  report a sample size it does not have. The harness does not send one today and
  must not acquire one. Any future provider-seed flag is prohibited in battery
  cells and would require an amendment.
- **Sampling pin:** `temperature = 1.0`, `top_p` never sent; on a parameter
  rejection the pin is dropped for that call and the record's
  `sampling.temperature` is null (v1.0 A3 implementation addendum). Mixed cells
  stay visible per record.
- **Cell identity:** provider, model pin, task, injector or case, arm, N, seed,
  evidence hash, harness commit, adapter commit (baseline arms), prompt variant,
  strict flag, date. All of it lands in the JSONL filename and the run record.
- **Order:** the arms of one comparison run inside a 24-hour window (§2.7).

---

## 6. Exclusions, telemetry, provenance

- v1.0 §7 exclusion rules apply unchanged: only infrastructure failures may
  exclude a response, each logged with its cell and reason; refusals, empty tool
  calls and weird-but-parseable outputs are data. Transport double-failures
  carry an error marker and appear as per-cell error counts in every table
  (A3.9b).
- New in Round 2: **adapter and toolkit failures are data, not exclusions**
  (§2.7), counted in their own column.
- Telemetry required on every Round-2 cell: latency, token usage and cost, calls
  per response, `rejection_kinds` composition, claims per response, abstention
  rate, error and empty counts.
- Manuscript tables come from `scripts/make_tables.py` over committed JSONL via
  `scripts/independent_scorer.py`, each table carrying the git commit and the
  scorer cross-check disagreement count (A3.9a). Baseline arms go through the
  same scorer; no toolkit's self-report is an outcome.
- Every Round-2 block gets a run record at
  `paper/ablation_live_<provider>_<date>_<block>.md` with harness and adapter
  commit pins, before the next block starts.

---

## 7. Prediction register

Every prediction in this plan with the outcome that rejects it. Nothing here is
revised after data; a wrong prediction is a result and is reported as one, the
way the iid retry model already was.

| # | Prediction | Rejected by |
| --- | --- | --- |
| H7 | Guardrails stock loop leaves the composite rate within 2pp of the bare floor (k >= 6 at N = 200) | 0/200 on all channels where L1 > 3%, or composite more than 10pp below L1 with the interval excluding -10pp |
| H8 | Static strict schema leaves entity fabrication within 2pp of the bare floor | 0/200 on entity while the matched L1 > 3% |
| H9 | Decode-enforced call-time enum ties the contract on entity (0 vs 0, inside delta) | k >= 1 on entity in `A-STRICT-ENUM`, after §3.5 adjudication |
| H10 | On an activated non-entity channel, decode enforcement shows k >= 6 while the contract shows 0 | the activated channel reads 0 in `A-STRICT-ENUM`, or no channel activates (X2 void, LC-3) |
| H11 | Our checks inside the Guardrails loop match the contract on all channels inside delta | any channel difference outside delta with the Newcombe interval excluding 0 |
| H12 | Wording effects do not transport: 20pp movement on some variant, or a different variant ordering, between providers | all six variants within 10pp across providers and identical orderings |
| H13 | Every contract arm reads 0 on entity and value on every provider | any k >= 1, treated as a defect under §3.5, not as a rate |
| H14 | The value channel activates (k >= 5, N = 200, unguarded) in at least one admissible pressure cell | every §4.3 cell reads k <= 4 (published as the §4.6 boundary) |
| H15 | The omission channel activates under low anchor salience | every §4.3 cell reads k <= 4 with abstention below 80% |
| H16 | The post-strip completeness re-check fires at least once on live data | no such case in the whole battery (reported as an unexercised code path) |
| C-S | No generated (evidence, response) pair passes Tier B carrying an off-evidence entity or an out-of-tolerance claim | one such pair found by the §1.6 property search; C-S withdrawn or narrowed |

---

## 8. Definition of done for Round 2

1. This file is committed before the first Round-2 live call.
2. The `layer3_bare` arm, the baseline adapters and the strict-mode paths are
   committed and mock-tested, with their commits pinned in the run records.
3. Each pressure instance is frozen in §9 with its offline verification output
   before the first live call on it.
4. Every executed cell has a run record, JSONL logs and independent-scorer
   output in the repository.
5. Every claim in the manuscript traces to a cell record, with its interval
   recomputed from raw k/N, and every 0/N cell carries the §1.4 wording and the
   analytic-or-empirical label.
6. Every losing condition that occurred is stated in the results, including in
   the abstract where it bears on a headline.

---

## 9. Amendments log (Round 2)

Amendments are appended here with a date and a reason, BEFORE the run they
affect. Amendments to v1.0 continue to belong in v1.0 §8; this log covers only
the Round-2 additions defined in this file.

- *(none yet, 2026-09-14)*

Slots reserved, to be filled before the corresponding first live call:

- **B1 — pressure-instance freeze (§4.2).** Instance specs, seeds and the
  offline reachability verification output for every value-channel and
  omission-channel instance beyond the already-frozen #14.
- **B2 — baseline adapter freeze (§2.1).** Guardrails and NeMo versions,
  configurations, reask budgets, the exact validator wiring for `A-GR-OURS`, and
  the capability-table entries for checks a toolkit cannot express.
- **B3 — strict-mode lane confirmation (§2.1, §3.1).** Which providers support a
  switchable strict flag on the tool path at run time, including the DeepSeek
  beta base-url condition, and which arms are therefore unavailable per
  provider.
- **B4 — panel additions (§3.1).** Any provider added to the Round-2 panel, with
  its reason, logged before its first cell.
