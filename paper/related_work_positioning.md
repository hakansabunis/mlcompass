# Related Work positioning package — INISTA_Paper.md

**Sweep date:** 2026-09-14 · **Scout:** lit-scout · **BibTeX:** `paper/references_new.bib`

Every identifier below was fetched and confirmed on the sweep date. Works that
could not be resolved are in the **UNVERIFIED** section at the bottom and are
deliberately absent from the `.bib`.

**Headline:** nothing found scoops the paper outright. No SCOOP ratings were
issued. But two of the manuscript's claims need narrowing before arXiv, and one
of them — the repair asymmetry — is not what the team believes it is.

---

## 0. New since the last sweep (last ~6 weeks: 1 Aug – 14 Sep 2026)

Five items are genuinely new and three of them touch load-bearing claims.

| # | Work | Date | Why it matters now |
|---|------|------|--------------------|
| 1 | **CHyD** — Guaranteeing Faithful Evidence Extraction in Speculative RAG, `arXiv:2609.10046` | 9 Sep 2026 | The only other 2026 paper claiming a **hard faithfulness guarantee** rather than a rate reduction. Direct competitor on the prevention-vs-mitigation rhetoric. |
| 2 | **A Removal Based Approach to Improve LLM Faithfulness at Test-Time**, `arXiv:2609.04343` | 3 Sep 2026 | Uses the **exact soundness/incompleteness vocabulary** of Prop. 2 and takes a removal-based repair. Does not prove the asymmetry, but a reviewer will raise it. |
| 3 | **CONTINUITY** — Security-Context Contracts for Composable LLM Agent Controls, `arXiv:2609.05269` | 4 Sep 2026 | Assume-guarantee contracts composed across agent boundaries. This is the manuscript's own stated future work ("contract composition along multi-step agent chains") already partly occupied. |
| 4 | **Agent Safety Should Be a Runtime Contract**, `arXiv:2608.11274` | 11 Aug 2026 | Position paper that claims the phrase *runtime contract* for agent safety and runs a **false-completion audit** — agents claiming work they did not do. Better field evidence for the manuscript's motivation than the SG/KR paper (see §3.1). |
| 5 | **ProvenanceGuard** revision, `arXiv:2606.18037v2` | 27 Aug 2026 | v1 was June; the August revision keeps it live. Claim-level verification of MCP tool output with allow/block. |

Also revised in-window: **HALO / "Zero Hallucination, by Construction"**
(`arXiv:2607.17883`, v2 on 1 Sep 2026) — an architecture position paper with no
measurements, but it owns the phrase *by construction* in this area now.

Nothing in the window does call-time enum binding to a deterministic evidence
dictionary, numeric claim-value checking with tolerance, or completeness
checking against a ranked anchor. Those three remain unoccupied.

---

## 1. The three judgements

### 1.1 Does the "no prior work packages … as a single contract" claim survive?

**It survives as literally written. It should not be left as written.**

Taken as a four-way conjunction — evidence-bound domains *and* claim-level value
checks *and* completeness checking *and* provider-independent enforcement — the
claim is true. I could not find any work doing all four. But the conjunction is
now propped up by only two of its four conjuncts, and reviewers punish exactly
this shape of claim.

Component by component, against what I verified:

| Component | Occupied? | By whom |
|---|---|---|
| Evidence-bound domains at call time | **Partly** | CHyD `2609.10046` constrains decoding to spans present in retrieved evidence — same idea, decode level, span granularity. EG-VAR `2607.12650` binds to per-source lifts but **committed at curation time, audited offline** — not call time. FAX `2605.27879`: no schema binding at all. |
| Claim-level **value** checks (numeric, with tolerance) | **No** | Confirmed absent in FAX (categorical refuted/corroborated/inconclusive, no tolerance), EG-VAR (kernel type-checks proof structure, not numeric precision bounds), ProvenanceGuard (NLI support + token alignment, not numeric equality), Narration Gap (binary SAT verdicts). |
| **Completeness** / anchor coverage of a ranked candidate | **No** | Absent in FAX (no mandatory coverage requirement), EG-VAR (explicitly *trades away* internal completeness for soundness and proof economy), ProvenanceGuard. Narration Gap uses "complete" in the **monitor-completeness** sense — detecting every flipped conclusion — which is a different predicate. Do not let a reviewer conflate the two; say so in the text. |
| Provider-independent enforcement | **Yes — occupied** | EBTE `2607.25364` is literally "Server-Verified Action Claims **Without Trusting Model Rationales**". Narration Gap's enforcer is provider-independent by bypassing the narrator. Guardrails AI [18] and NeMo Guardrails [19] already ship it. |

**Recommendation.** Keep the claim, but narrow it and cite into it rather than
around it. Concretely:

- Drop *provider-independent enforcement* from the list of novel components. It
  is not novel and the manuscript already half-concedes this in the [18], [19]
  sentence. Reframe it as a **design requirement** the setting imposes, not a
  contribution.
- Lead the claim with the two conjuncts that are genuinely unoccupied: **numeric
  claim-value checking against a deterministic producer** and **completeness
  against a ranked anchor**.
- Add one sentence distinguishing **call-time** enum binding from EG-VAR's
  **curation-time** lifts and CHyD's **decode-time** span constraint. That
  distinction is real, defensible, and currently unstated.
- Keep "over a decidable faithfulness class". That qualifier does real work and
  nothing I found competes with it; EG-VAR mentions decidability only as a
  trade-off it makes, not as a class it defines.

Suggested replacement for the sentence at II ¶3:

> To our knowledge no prior work binds the narrator's admissible value domain to
> a deterministic evidence producer *at call time* and then re-verifies both the
> numeric content of each claim and the coverage of the top-ranked candidate in
> code the provider does not control. Contemporary systems occupy the components
> separately: [CHyD] guarantees verbatim spans at decode time; [EG-VAR] attests
> claims against lifts audited offline at curation time; [FAX] decomposes and
> verifies claims but assigns status by LLM and carries no numeric or coverage
> channel; [EBTE] verifies typed *action* claims server-side without trusting
> model rationales.

### 1.2 Is the repair asymmetry (Prop. 2) novel?

**No. The belief is wrong, and this is the most important finding of the sweep.**

The asymmetry — universal/"nothing may be present that is not licensed"
constraints repairable by deletion, existential/"something must be present"
constraints not repairable by deletion — is a **classical result in database
repair theory**, thirty years older than the manuscript. The canonical reference
is Chomicki and Marcinkowski, *Minimal-change integrity maintenance using tuple
deletions*, Information and Computation 197(1–2):90–121, 2005
(`doi:10.1016/j.ic.2004.04.007`, verified via Crossref). The standard statement
in that literature is that denial constraints can only be repaired by deleting
tuples, whereas insertions are required for inclusion dependencies and, in
general, tuple-generating dependencies. That is Prop. 2's shape exactly, with
*sound response set* in place of *denial constraint* and *anchor coverage* in
place of *inclusion dependency*. Belief revision has the same structure in the
AGM contraction-versus-expansion distinction.

There is also a behavioural precedent in the immediate neighbourhood: **FAX
already does strip-vs-flag** — corroborated claims retained, refuted claims
removed, inconclusive claims omitted — without formalising why. And
`arXiv:2609.04343` (3 Sep 2026) defines unsoundness as "the explanation cites
factors that did not influence the model's answer" and incompleteness as "the
explanation omits factors that influence the answer", then pursues a
removal-based repair. It proves no asymmetry theorem, but it is two weeks old
and uses the manuscript's vocabulary.

**What actually survives as the manuscript's own.** Not the asymmetry. The
*safety* argument layered on top of it:

1. In a database there is no narrator, so insertion is merely a different repair
   operation. In narration, insertion **fabricates attribution** — it puts words
   in the narrator's mouth, which is the very failure the contract exists to
   prevent. Deletion is therefore not just the cheaper repair but the **only
   admissible** one. Database repair theory has no analogue of this step because
   it has no speaker.
2. The corollary in the same proposition — that substituting an abstention is
   *also* unsafe because it attributes an epistemic stance the model did not
   take — is, as far as I can find, unstated anywhere. It is small, but it is
   genuinely yours.

**Recommendation.** Recast Prop. 2 as a *transposition with a safety argument*,
not as a new formal result, and cite Chomicki & Marcinkowski in the proposition
itself. Concretely: rename the consequence from a formal claim to an
observation; add "This asymmetry is the narration-side instance of a classical
fact about database repair [chomicki2005minimal], where denial constraints admit
deletion-only repairs while inclusion dependencies require insertion. What the
narration setting adds is that insertion is not merely a different repair but an
unsafe one: it attributes content to a speaker who did not produce it." Then let
Prop. 1 and the measurements carry the contribution weight.

Left as-is, Prop. 2 is the single most likely thing to draw a hostile review from
anyone with a database-theory background, because it is presented as a discovery
and it is not one. Fixed as above, it is a clean and defensible framing move.
Contribution 1 in §I should be reworded to match — the phrase "the two formal
facts the design rests on" currently implies both are the paper's own.

### 1.3 Cross-check of `paper/references_verified.md`

**All 53 entries resolve. No entry needs to be pulled.** Checked on 2026-09-14:

- 29 DOIs resolved against the Crossref API; every returned title matched the
  entry.
- 23 arXiv IDs resolved via the Semantic Scholar batch endpoint; every returned
  title matched.
- 10 publisher/anthology/proceedings URLs (PMLR v235 and v267, NeurIPS
  proceedings, ACL Anthology, JSE) all returned HTTP 200.

Two notes, neither fatal:

1. **`[guardagent]` — title drift.** The entry reads "GuardAgent: Safeguard LLM
   Agents **by a Guard Agent** via Knowledge-Enabled Reasoning". The current
   record for `arXiv:2406.09187` reads "GuardAgent: Safeguard LLM Agents via
   Knowledge-Enabled Reasoning". The title changed between arXiv versions. Pick
   the version you are citing and match it.
2. **`[bfcl]`** is cited as "Proc. ICML, PMLR vol. 267, 2025, pp. 48371–48392"
   on the strength of a PMLR URL only. The URL resolves; the volume/page detail
   was not independently re-derived in this sweep. Low risk, worth one glance
   before submission.

---

## 2. Positioning lines, grouped to the manuscript's paragraph structure

Format per work: **identifier** · what it does · how this manuscript differs ·
**THREAT**.

### ¶ "Hallucination and faithfulness" (currently [8], [9])

- **`arXiv:2510.05116`** (Xu, 29 Sep 2025) · Argues hallucination is inevitable
  for LLMs under the open-world assumption, and separates correctable from
  structurally unavoidable cases, noting that under a closed-world assumption
  hallucination may be mitigated. · This manuscript supplies the constructive
  complement the paper gestures at: a concrete, shipped narration class that is
  closed by construction, where the mitigable case is not just theoretically
  admitted but eliminated. · **CITE-ONLY** — and it is an asset, not a threat.
  It converts "evidence-closed" from a convenience into a theoretically
  motivated class. Cite it in the *first* Related Work paragraph, not late.
  **Title correction:** it is "Hallucination is Inevitable **for LLMs with** the
  Open World Assumption", not "under the Open World Assumption".

- **`arXiv:2609.04343`** (Luo et al., 3 Sep 2026) · Improves LLM self-explanation
  faithfulness at test time by removing concepts the explanation did not credit,
  framed explicitly along unsoundness and incompleteness. · Different object
  (a model's own reasoning, not a deterministic tool's measurements) and
  different mechanism (intervention on the input rather than repair of the
  output) — it sidesteps the repair asymmetry instead of confronting it. ·
  **REWORD** — cite it where Prop. 2 is stated, or a reviewer will assume you
  missed a two-week-old paper using your vocabulary.

### ¶ "LLM agents and tool use" (currently [5], [6], [7])

- **`arXiv:2606.04990`** (Wang et al., 3 Jun 2026) · Survey defining execution
  provenance as the typed graph of an agent execution and evidence tracing as
  its projection onto evidence-support relations. · This manuscript enforces at
  one boundary at call time rather than reconstructing support relations from
  traces after the fact. · **CITE-ONLY** — a cheap, strong anchor for "this is a
  recognised problem area", and it costs one clause.

- **`arXiv:2608.11274`** (Ng et al., 11 Aug 2026) · Position paper arguing agent
  safety belongs in a runtime contract with a preventive face and an evidential
  face, backed by an incident survey and a **false-completion audit** of agents
  claiming work they did not do. · Argues the case at the level of a research
  agenda; this manuscript instantiates one contract, in production, with live
  per-channel measurements. · **REWORD** — it claims the phrase *runtime
  contract* in agent safety. Either cite it as independent convergence on the
  framing (the stronger move) or avoid implying the framing is yours. Its
  false-completion audit is also better motivating evidence than the SG/KR
  paper — see §3.1.

- **`arXiv:2609.05269`** (Zheng & Yang, 4 Sep 2026) · Assume-guarantee contracts
  carrying authenticated security context across agent component boundaries, so
  individually correct controls compose end-to-end. · Security-context
  composition, not narration faithfulness; the manuscript's guarantee is about
  what may be *said*, not what may be *done*. · **CITE-ONLY** — but cite it in
  §VI where you propose contract composition along agent chains, or that
  future-work item reads as unaware of the nearest existing answer.

- **`arXiv:2606.26524`** (Li et al., 25 Jun 2026) · Runtime enforcement of
  natural-language behavioural policies from agent skill specs, compiled to SMT
  constraints over finite execution traces. · Enforces permission and value-flow
  policy over traces; enforces nothing about the truth of a narration. ·
  **CITE-ONLY**.

- **`arXiv:2602.22302`** (Bhardwaj, 25 Feb 2026) · Design-by-contract for agents:
  preconditions, invariants, governance, recovery, with a runtime assertion
  library. · Contracts over agent behaviour generally; no evidence binding, no
  faithfulness predicate. · **CITE-ONLY**, and optional. Single-author,
  non-peer-reviewed, marked patent-pending. Include only if you want breadth in
  the "contract" framing paragraph.

### ¶ "Constrained and structured generation (closest prior art)" (currently [12]–[16], [18], [19])

This is the paragraph that must change most.

- **`arXiv:2609.10046`** (Signé et al., 9 Sep 2026) · CHyD repurposes speculative
  decoding to enforce hard constraints restricting generation to contiguous
  spans present in the retrieved context, guaranteeing that any quoted span
  appears verbatim in the source. · Same ambition — a hard guarantee, not a rate
  reduction — but at decode time, on span extraction, and therefore requiring
  logit access the manuscript's setting does not have; no numeric value channel,
  no coverage requirement, and the evidence is retrieved documents rather than a
  deterministic measurement dictionary. · **REWORD**. This cuts two ways and the
  manuscript should say so. It weakens any implication that a hard faithfulness
  guarantee is unprecedented — but it *strengthens* the §II argument that
  decode-time enforcement is unavailable behind a commercial API, because CHyD
  is precisely what you cannot run there. Use it as the worked example of the
  constraint you are working around.

- **`arXiv:2607.12650`** (Ren, 14 Jul 2026) · EG-VAR makes a Lean 4 kernel the
  sole minter of "Verified" claims via tool-attestation axioms, so verified
  outputs trace to attested tool calls and kernel-checked inference. · Verified:
  the lifts from data to formal claims are **committed at curation time and
  audited offline by the source curator**, not bound to evidence at call time;
  evaluation is TableBench numerical reasoning, not pipeline tooling; no
  tolerance-based numeric comparison; no coverage check; it explicitly **trades
  away internal completeness** for soundness and proof economy; no repair or
  deletion semantics — unsound output surfaces as abstention. · **REWORD** —
  the manuscript's existing distinguishing line is accurate but under-specified.
  Say *curation-time* versus *call-time*; that is the crisp difference, and it
  is verifiable from their §on lifts.

- **`arXiv:2605.27879`** (Kim et al., 27 May 2026) · FAX decomposes an
  explanation draft into atomic claims, tests each against faithful tools, and
  retains corroborated / removes refuted / omits inconclusive claims before the
  answer reaches the user. · Verified: decomposition **and** status assignment
  are LLM-performed, not deterministic code; no JSON-schema enum constrains
  emission at call time (prompts are natural language); no numeric tolerance —
  verdicts are categorical; no mandatory coverage of a top-ranked item; the
  setting is open-world RL. · **REWORD**. Two corrections to the manuscript's
  current handling. (a) Calling FAX "post-generation filtering" is not quite
  right — it gates before the user sees the answer, same as Tier B; the real
  difference is *who decides*, an LLM versus plain code, and that is the
  stronger point anyway. (b) FAX's retain/remove/omit **is** the strip-vs-flag
  asymmetry in behaviour. The manuscript can still claim the formal account, but
  not the observation.

- **`arXiv:2607.25364`** (Zhu & Wang, 28 Jul 2026) · EBTE converts free-form
  model rationales into typed action claims and validates them server-side
  against held facts — intent, policy, payload, tool, risk, provenance,
  freshness — permitting execution only on match. · Verifies **action** claims
  (what the agent proposes to do) against policy state *before* execution; this
  manuscript verifies **content** claims (what a tool measured) against a
  measurement dictionary *after* generation. Different object, opposite
  direction along the loop. · **REWORD** — this is the strongest new challenge
  to "provider-independent enforcement" as a novelty component, and the reason
  §1.1 recommends dropping that conjunct. Cite it explicitly; do not let a
  reviewer find it first.

- **`arXiv:2606.18037`** (Alvarez et al., 16 Jun 2026, rev. 27 Aug 2026) ·
  ProvenanceGuard consumes captured MCP traces, decomposes answers into atomic
  claims, routes each to source-specific evidence, checks support with NLI and
  token alignment, compares stated against routed attribution, and returns
  per-claim verdicts plus an answer-level allow/block. · Targets cross-source
  conflation — a claim supported *somewhere* but attributed to the wrong source
  — using probabilistic NLI support; the manuscript's single-source setting
  makes attribution trivial and instead checks numeric equality and coverage
  deterministically, so its verdicts are exact rather than model-scored. ·
  **REWORD** — closest existing work on claim-level verification of tool output
  and currently uncited. It belongs in the same sentence as Guardrails [18] and
  NeMo [19], as the claim-level rather than pattern-level member of that family.

- **`arXiv:2607.17883`** (Raduta et al., 20 Jul 2026, v2 1 Sep 2026) · HALO, a
  six-layer enterprise assurance architecture — grounded generation from
  approved content, constrained deterministic execution, multi-signal
  verification, calibrated abstention, traceability, drift monitoring — arguing
  zero hallucination is a system property, not a model property. · Same thesis
  at the slogan level; it is an architecture position paper with no quantitative
  evaluation, no formal guarantee, and no decidability argument, whereas this
  manuscript's whole claim is that it *measured* the shipped path. · **CITE-ONLY**
  — but note it now owns the phrase "by construction" in this space. The
  manuscript uses "by construction" three times. Keep the phrase, cite HALO
  once, and lean on the measurements as the differentiator.

- **`arXiv:2607.24539`** (Zhao et al., 27 Jul 2026) · Task-conditional
  faithfulness audit for multimodal LLMs in power-grid diagnosis: registers
  task-specific evidence requirements, compares self-reported against
  intervention-derived reliance, and runs evidence-gated correction and
  re-audit. · An audit-and-correct loop over *which evidence was used*; this
  manuscript enforces *what may be cited and asserted*, and does it at the tool
  boundary rather than by ablation. · **CITE-ONLY** — useful as an independent
  instance of "evidence requirements registered up front" in a different
  high-stakes domain, and it supports the §V-D transposition argument.

### ¶ "Code assistants and their reliability" (currently [10], [11])

- **`2025.acl-long.1480`** / `arXiv:2410.14748` (Maharaj et al., ACL 2025) · ETF
  extracts code entities by static program analysis and uses LLMs to map and
  verify them in generated summaries, with CodeSumEval (~10K samples) and 73%
  F1, localising the error within the summary. · Detection only, post-hoc, no
  prevention, no value channel, no completeness channel — and its F1 of 73%
  makes the manuscript's point: a detector that is itself a model leaves a
  residual, whereas a decidable check does not. · **CITE-ONLY**. One correction
  to the internal map: ETF is **not** deterministic end to end. Extraction is
  static analysis; the mapping and verification step is LLM-based. Do not
  describe it as "deterministic entity extraction plus membership testing" — a
  reviewer who knows the paper will catch it, and the corrected version is
  better for you anyway.

### ¶ "ML pipeline tooling" (currently [1]–[4], and [17] in §III)

- **`arXiv:2509.15971`** (Truong et al., 19 Sep 2025) · LeakageDetector 2.0, a VS
  Code extension detecting overlap, preprocessing and multi-test leakage in
  Jupyter notebooks, offering both a manual quick-fix and an LLM-driven
  guidance path. · Direct prior art for the leakage-detection tool, and its
  LLM-driven guidance path is an unguarded narrator over leakage evidence —
  exactly the surface this manuscript's contract governs. · **CITE-ONLY, but
  mandatory.** Currently absent from the manuscript and it should not be. It is
  the closest existing tool to `detect_leakage`, and the honest framing is
  strong: they ship the detector and an ungoverned LLM explanation layer; this
  paper governs that layer. Place it alongside [17] and the notebook static
  analysis work (`yang2022notebooks`, already verified).

### ¶ NEW — data-to-text ancestry (the manuscript currently has none)

The manuscript engages none of this and it must. Five entries are already
verified in `references_verified.md` (`wiseman2017challenges`,
`dhingra2019parent`, `parikh2020totto`, `gardent2017webnlg`, `chen2020logicnlg`).
Add one:

- **`arXiv:2102.08585`** (Liu et al., AAAI-21) · Entity-centric view of
  table-to-text faithfulness: metrics for table-record coverage and the ratio of
  hallucinated entities, plus training-side fixes. · This is the manuscript's
  problem, five years earlier, at evaluation and training time: their
  *hallucinated-entity ratio* is the entity channel and their *record coverage*
  is the completeness channel. The difference is that they measure and reduce,
  while an evidence-closed setting lets the same two quantities be driven to
  zero and held there by construction. · **CITE-ONLY**, but load-bearing for
  credibility. Suggested one-liner for the paragraph: *"The entity and coverage
  channels are not new — entity-centric table-to-text faithfulness has measured
  both for half a decade [liu2021entitycentric], [dhingra2019parent]. What
  changes in an evidence-closed setting is that both become decidable, so they
  can be enforced rather than scored."* Written this way the ancestry becomes an
  asset; left out, it is the paragraph a reviewer uses to reject.

### ¶ NEW — solver/verifier loops and the narration boundary

- **`arXiv:2606.19588`** (Huang & Deng, 17 Jun 2026) · Models LLM-solver
  pipelines as verified decision procedures and shows the solver's soundness
  guarantee is lost in the narration step; **Theorem 3.2**: "A sound and complete
  monitor exists iff a sound enforcer exists, and each is built from the other."
  Verified: threat model is **indirect prompt injection** by an adversary
  controlling untrusted context, the narrated object is a **binary SAT verdict**
  mapped to holds/fails, and the enforcer works by **removing the LLM from the
  final answer path entirely**. · The manuscript's threat model is explicitly the
  model's own propensity under an honest operator (§III-A rules injection out of
  scope); its narrated object is a structured evidence dictionary with numeric
  claims, not a binary verdict; and its enforcement cannot bypass the narrator,
  because the narration **is** the product. · **REWORD** — the manuscript's
  existing distinguishing sentence is correct and should be kept; what is
  missing is an explicit note that Theorem 3.2 concerns monitor/enforcer
  equivalence and does **not** overlap Prop. 1 (verification complexity) or
  Prop. 2 (repair semantics). State that, or a reviewer will assume overlap. The
  paper's own conclusion — that robustness "does not reach to the answer that
  the user finally reads" — is the single best external sentence for the
  manuscript's motivation.

### ¶ NEW — formal ancestry of Prop. 2

- **`doi:10.1016/j.ic.2004.04.007`** (Chomicki & Marcinkowski, Inf. Comput. 2005)
  · Minimal-change integrity maintenance restricted to tuple deletions;
  characterises which constraint classes admit deletion-only repair, covering
  denial constraints, functional and inclusion dependencies, and key/foreign-key
  constraints. · The manuscript has no *database*; it has a *narrator*, and the
  novel step is that insertion is not merely a different repair but an unsafe
  one, because it attributes content to a speaker who did not produce it. ·
  **CITE-ONLY — but it is the most consequential citation in this package.**
  See §1.2. Adding it costs one sentence and converts Prop. 2 from a
  vulnerable claim into a defensible transposition.

---

## 3. Corrections to the internal known map

These are errors in the scouting notes, not in the manuscript, but two of them
would have propagated into the paper.

### 3.1 `arXiv:2606.17114` is materially mischaracterised

The map records it as documenting "agents narrating tool states never observed
— the failure mode, evidenced in the field by government bodies." **It does
not.** The verbatim abstract is a Singapore/Korea AI Safety Institute joint
evaluation of **privacy and sensitive-information exposure** by tool-using
agents across 12 non-adversarial tasks, on five risk types: data awareness,
audience awareness, policy compliance, data minimisation, access-boundary
awareness. Its finding is that task success often coincides with data-handling
failures.

The only sentence bearing on this manuscript is a secondary qualitative note:
"Qualitative review also revealed **claim-action mismatches**, simulation-aware
behavior, user-simulator role reversal, and interpretation gaps in automated
judging."

Consequences:

- It **cannot** be cited as field evidence of narrators fabricating tool
  evidence. That would misrepresent the source, in a paper about fabrication.
- It **can** be cited, narrowly, for *claim-action mismatches observed by
  national AI safety institutes*, flagged as a qualitative secondary finding.
- There is a **terminology collision**: "data leakage" there means privacy
  exfiltration; in this manuscript it means ML train/test contamination. If both
  appear, disambiguate on first use.
- **Better substitute:** `arXiv:2608.11274` carries an explicit false-completion
  audit (31 non-contested core cases) of agents claiming work they did not do.
  That is much closer to the manuscript's failure mode and is properly
  citable for it.

### 3.2 Two title errors

- `arXiv:2510.05116` is "Hallucination is Inevitable **for LLMs with** the Open
  World Assumption" — not "under the Open World Assumption". Author: Bowen Xu,
  single author.
- `arXiv:2607.12650` (EG-VAR) is titled "**Evidence-Grounded Verified Agentic
  Reasoning**: A Path Toward Eliminating LLM Hallucination in Empirical
  Inference via Tool-Attested Kernel Proofs". Single author: Junyu Ren. Note it
  is a workshop paper (ICML 2026 TAIGR per its abs page), which is worth knowing
  when weighing how hard to push back on it.

### 3.3 One characterisation error

ETF is not deterministic end to end — see the Code-assistants entry above.

---

## 4. UNVERIFIED — do not cite, not in the .bib

- **"Adjudicating Artifact-Faithfulness Claims in Tool-Using LLM Agents: A
  Trace-Local Protocol"**, OpenReview `40wuXQMQRU`. Surfaced in search and looks
  directly relevant — the snippet describes agents that "call a verifier but
  still emit final artifacts that contradict its output". OpenReview served a
  bot-check page on every attempt, so title, authors, venue and status are
  **unconfirmed**. Worth one manual look in a browser before submission; if it
  is what the snippet suggests, it is the closest thing to a competitor found in
  this sweep and would need its own assessment.
- **AAAI-21 proceedings record for `liu2021entitycentric`.** The arXiv entry is
  verified and its comments field reads "AAAI-21 Camera Ready", but the AAAI
  volume and page range were not independently resolved. Cite the arXiv version,
  or resolve the proceedings page first.
- **Surfaced in search but never individually fetched** — titles and authors are
  therefore unconfirmed and none may be cited as-is: `2603.19532` (EvidenceRL),
  `2601.15322` (Replayable Financial Agents), `2603.02798` (Guideline-Grounded
  Evidence Accumulation), `2606.16118` (Know Your Limits), `2608.18565`
  (SemaPLC), `2609.12017`, `2608.11033`. On the abstracts as summarised, none
  appeared to threaten a claim; they are listed only so the gap is on the record.
- **`arXiv:2607.28225` (FaithEyes)** was fetched and resolves — Haoqing Wang,
  Xingrun Xing, Wei Xia, Ziheng Li, Yehui Tang, 30 Jul 2026 — but it is
  deliberately **excluded** from the `.bib`. It concerns whether a VLM's tool
  calls are *useful*, not whether its narration is *faithful*. Different
  predicate; including it would pad the paragraph without earning anything.

---

## 5. Two flagged risks outside the brief

1. **`arXiv:2603.03538` has a title that changed between versions** — the current
   abs page reads "Verify to Amplify: Improving Reasoning via Learned
   Chain-of-Thought Verification" while v3 is served as "Online Learnability of
   Chain-of-Thought Verifiers: Soundness and Completeness Trade-offs". It is in
   the `.bib` with a warning note. It is optional and only worth citing if you
   want to mark that *soundness* and *completeness* are used with a different
   meaning in the verifier-learning literature (there they name verifier error
   types, not response properties). If §III-C keeps both senses in play, one
   clause disambiguating them is cheap insurance.

2. **Narration Gap's "complete monitor" versus the manuscript's "completeness
   channel"** are different predicates sharing a word, in the paper the
   manuscript names as its closest by framing. Worth one explicit sentence.
