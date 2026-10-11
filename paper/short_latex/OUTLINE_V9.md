# V9 outline: Q1 empirical SE format (ARS outline-only mode, 2026-10-11)

Config. Venue type: Q1 empirical SE journal (TSE/EMSE layout), IEEEtran, 10 pages
of text + 2 of references. Rules: abstract <= 200 words in five parts, no
dashes/hyphens/apostrophes in the abstract, no "it", no "engineer", title
unchanged, no new experiments or analyses. Every number already appears in
V8.2 or in the extended version and was checked against the records.

## Structure and paragraph plan

**Abstract** (five parts, 199 words): problem; prevailing approaches and their
limits (decode control the caller lacks, provider schemas whose enforcement the
caller cannot verify, pooled entity and number checks); proposal; results; contributions.

**I. Introduction** (~1.0 p)
1. Context: LLMs narrate deterministic tool output; narration borrows authority.
2. The motivating failure in one paragraph (r2 in 42 %; printed as verified
   evidence), pointing to Section II.
3. Gap: misfiling vs invention; pooled checks pass misfiling by construction;
   where existing defenses act (Table VIII in Related Work).
4. Idea: admissibility per field, numbers keyed to (entity, statistic), bound at
   call time; Tier A requests, Tier B checks; loop is ordinary validate-and-reask.
5. RQ1 to RQ3.
6. Findings in brief (1,772 of 1,831 only evidence items; field description;
   stale list to invention; contract zero; parity on leakage names; A11 residue).
7. Contributions (i)-(iii), one framing clause (contract plus empirical analysis,
   not a new algorithm).
8. Scope in two sentences (non-claims collected once; details in Threats).
9. Organization sentence.

**II. Motivating Example** (~0.45 p) NEW
- Fig. 2 (listing, one column): excerpt of E for the reference task; answer 32
  of the floor run of 2026-09-19 (cited columns r2, log_target_v2,
  near_target_proxy; two exact correlation claims); what three checks return.
- Text: pooled check passes (r2 is a name of E); data-frame check rejects r2 but
  would admit an unlisted real column and checks no number; contract flags r2
  under (C1), names r2 in the retry, strips after the budget. Second channel:
  the two leak candidates differ by 0.00026, inside tau, so a swapped value
  would pass (C2): the example shows both what the check sees and its limit.

**III. Evidence-Bound Contract** (~1.1 p)
A. Evidence and admissibility: Def. 1 evidence-closed; Def. 2 output conditions
   (C1) entity, (C2) value; Def. 3 anchor-coverage warning (C3), defined only
   when E has an anchor; misfiled / unlisted / invented.
B. Two tiers: Tier A flat, independent enumerations (a request unless the
   provider enforces the schema); Tier B check, named retry <= 2, strip, abort
   on no admissible verdict, flag (C3) after strip. Fig. 1.
   Proposition 1 with assumptions (fixed E; payload parsed into fields; every
   structured claim through the same filter; an exception in the check
   propagates, so nothing unchecked is rendered). Proof sketch: the filter keeps
   exactly the items that meet the predicates of (C1)-(C2).
   Guarantee vs empirical reliability vs end-to-end correctness.
C. Implementation: ~40-line specification; Instructor port as feasibility; cost
   linear in the sizes of the answer and E; tests and falsifier defects.
D. What the contract does not cover: closeness not provenance (tau
   neighbours, nine values in two crowded answers); verdict; free text; trusted
   detector; unguarded paths; prompt injection.

**IV. Study Design** (~1.4 p)
A. Research questions, each with its motivation.
B. Subjects: tasks; narrators and sampling.
C. Configurations in three groups (Table III groups): diagnostic controls
   (floor, T=0, stock Guardrails, typed schema without list, stale list);
   interface fixes (shipped prompt, its first three rules, field description);
   mechanisms bound to E (call-time enumeration +/- strict flag, choice list,
   strong configuration without check); the contract (our checks in Guardrails,
   Tier A+B, Tier B alone, as shipped, strong + Tier B). Strict-flag note.
D. Measures: one scoring table; violation; flagged; exclusive kinds vs
   non-exclusive indicators (1,807 contain misfiling; 1,772 use only items of
   E; pooled acceptance stated on 1,772 only); claims matching E; tolerance;
   intervals; two scorers; seventeen instrument faults.
E. Registration and analysis: plan; confirmatory; materiality rule as
   registered; verifier zero; void/unrun items by content; A7; deviations by
   weight. Table II with descriptive names first, codes in parentheses.

**V. Results** (~3.4 p), each RQ: Motivation (1 sentence), Approach (1-2
sentences), Results (paragraphs), boxed Finding.
A. RQ1: misfiling dominates; what violations introduce (Table IV kinds, main
   corpus excluding the 3,200 follow-up responses); semantic reading; three
   quantities; what removes the failure. Finding 1.
B. RQ2: rates (Table III); same-domain parity on leakage names; diagnostic
   controls (stock loop, typed schema, stale list with A1 rejected, A14);
   coverage map (Table V, by-definition verdicts with observed counts in
   brackets); verifier in the best configuration (A11); cost; content kept
   (no reduction in the measured count of claims matching E); why Tier B.
   Finding 2.
C. RQ3: profiling task; Table VI; misfiled values (exploratory); how the tiers
   separate (flat enumerations); class balance. Finding 3.

**VI. Discussion** (~0.6 p): design principles; design alternatives (strip
without retry; host filling; enforced strict mode, both unmeasured; tool-server
boundary); beyond leakage.

**VII. Threats to Validity** (~0.55 p): construct, internal, external, conclusion.

**VIII. Related Work** (~0.75 p): faithfulness to a finite record; where
constraints act (Table VII taxonomy); narration under a check; positioning.

**IX. Conclusion and Future Work** (~0.25 p): summary; future work
(enforced strict schema, host-filled values, a second producer, independent
raters, per-statistic tolerance).

Acknowledgments, AI disclosure, Data Availability.

## Evidence map (claim -> source)

| Claim | Source |
|---|---|
| 42.0 % floor; four runs 42.0/43.0/40.5/49.0, p = 0.34 | Table III; drift_tests.json |
| 1,831 violating, 1,829 flagged, 1,772 misplaced, 57 extrinsic | Table IV; unified_summary.json |
| 1,807 of 1,829 flagged contain misfiling (non-exclusive) | unified_scoring.csv |
| 81 of 200 list r2, 73 beside all ten columns | 2026-09-19 floor records |
| answer 32 of 2026-09-19 floor | scripts/runs/2026-09-19_freetext |
| leak candidates 0.998993 / 0.998736 (gap 0.00026) | evidence_618bd74a.json |
| field description 49.0 -> 0.5 %; shipped 0/200; rules 7.0 %; T0 2.0 % | Table III |
| stale list 145/200 invented; A1 rejected (76.0 vs 27.5 %) | Table III; plan A1 |
| A14 143, 2, 0, 3, 0; p < 1e-76; p = 0.25 | a14_results.json |
| A11 10/600 vs 0/600; p = 0.002; [0.7, 3.0]; 9 of 10 outside enums | a11_results.json |
| 18 of 200 named absent anchor, 5 with list columns | stale-list crowded records |
| nine neighbour values pass at tau = 0.005 | A-GR-CHOICES crowded answers 87, 119 |
| tau counts 25 / 24 | tau_sensitivity.json |
| 17 us median, 41 us p95; 3.3 s vs 2.8 s | verifier_timing.json |
| 170 claims before and after retry | retry_content.json |
| profile Table VI rows | unified_scoring.csv |
| judges A10, A12 failed; A13 55/55, 73/79, 40/40 | plan A10-A13 |

## Float plan

| Float | Content | Place |
|---|---|---|
| Fig. 1 | contract flow (existing PNG) | III-B, one column |
| Fig. 2 | worked example listing | II, one column |
| Table I | registered tests, descriptive names | IV-E, one column |
| Table II | arms in groups, rates, claims matching E | V-B, two columns |
| Table III | kinds of violation, main corpus | V-A, two columns |
| Table IV | coverage map, verdicts + observed counts | V-B, two columns |
| Table V | profiling task | V-C, one column |
| Table VI | taxonomy of constraint loci | VIII, one column |

## Page budget (text)

I 1.0 | II 0.45 | III 1.1 | IV 1.4 | V 3.4 | VI 0.6 | VII 0.55 | VIII 0.75 |
IX 0.25 | back matter 0.5 = 10.0 pages, floats included in their sections.
