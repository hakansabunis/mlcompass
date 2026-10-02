# Where this stopped — 2026-10-02 (late night)

**V2.7: DONE** (the V2.6 review: EMSE minor revision, perhaps accept with minor
changes; TSE between minor and major; no new experiment asked for).

- Title: "Misfiling, Not Just Invention: Failures in LLM Narration of
  Structured Evidence" (the review's safer option: the paper no longer says
  that invention is absent; it says misfiling can dominate). Changed in the TSE
  source, the EMSE port, README and CITATION.cff.
- Contribution 4 is the auditable package; the seventeen faults follow the list
  as evidence of auditability, not as a contribution.
- Abstract says "false relations" (the taxonomy's term); 249 words.
- RQ1 and Threats say why A7 was not run (the authors decided not to label; two
  authors' labels would not have been independent anyway) and that the sheet is
  kept, key sealed, for independent raters.
- Claim-strength pass: one general "narrators" phrasing scoped to our runs.

**V2.6: DONE** (the V2.5 review: EMSE minor revision, TSE major revision risk;
no new experiment asked for). Editorial pass only, no number changed:

- Claim narrowed (abstract, introduction, contributions, conclusion): misfiling
  can dominate when an evidence-closed narrator fails, and an aggregate
  hallucination rate hides it; no general claim about LLMs.
- The A13 reading is worded as a classification by the one judge that passed,
  not as "misleads".
- RQ2/3 opens by separating the (C1)-(C2) guarantee (a design property) from
  the empirical questions (implementation, cost, added benefit = A11).
- Corrections open with the immutable records and the two independent
  scorers' full agreement on entity, misfiled and out-of-evidence flags.
- Main text about 10 % shorter (prose 16,379 -> 14,942 words; with tables
  21,819 -> 19,673). Nothing deleted: Appendix A "Audit Trail" (tables A1 and
  A2) holds the other registered items (A3 R3/R4, A7, A9.2, A10, A12), the
  failed judges, the rejected attempts, fault details and patterns, the
  free-text scan, verdict counts, drift, the offline deletion analysis and the
  downstream benchmark. Ports map `\appendices` to `\appendix`.
- Neutral build 26 pages (main text ends on page 20); IEEE and EMSE build; no
  overfull boxes or undefined references; both table series in citation order.

**V2.5** Venue-neutral two-column build (`paper/port_to_neutral.py` ->
`paper/neutral_latex/main.pdf`, 25 pages, copied to Desktop/MLCompass/V2.5.pdf)
for the next simulated review; the IEEE (`paper/tse_latex`) and EMSE
(`paper/emse_submission`) builds come from the same source. No overfull boxes,
no undefined references, tables numbered in citation order, abstract 250 words,
no "it". New since V2.4:

- **A11 (registered, interleaved, N = 600 per arm):** the verifier lowers
  delivered violations in the best configuration found, 10/600 -> 0/600
  (Fisher p = 0.002, difference 1.7 points, Newcombe [0.7, 3.0]), same content
  (29.4 correct claims per response). All 11 rejected attempts confirmed.
- **A12:** two LLM judges (gpt-5.4-mini, deepseek-chat) fail the validity check.
- **A13 (last automatic substitute for A7):** gpt-5.5 passes (55/55, 73/79;
  64/65 after fault 17) and reads every misfiled `r2` as a false relation
  (40/40). New Table "semantic audit" (`scripts/audit_table.py`).
- **Fault 17:** "a number E carries" was read from the dump without the derived
  absolute correlations; fixed in the kinds and the audit strata (13 sweep
  responses extrinsic -> misplaced; 15 of 17 S7 items re-assigned). Verdicts of
  all five judges unchanged; the sheet still rebuilds byte-identically.
- Spend: DeepSeek balance 3.20 USD (A11 about 1 USD); OpenAI about 3.7 USD (A13).

## Next

1. **Title** — changed again 2026-10-02 (V2.7) to "Misfiling, Not Just Invention: Failures in LLM Narration of Structured Evidence"; before that shortened to "Misfiling, Not Invention: How LLM Narrators of Structured Evidence Fail"
   (TSE and EMSE manuscripts, README, CITATION.cff). On 2026-10-01 it had
   become "Misfiling, Not Invention: How an LLM Narrator of Machine-Learning
   Pipeline Evidence Fails, and What an Evidence-Bound Contract Checks",
   after the V2.1 review read the paper as an empirical diagnosis rather
   than a mechanism; that title read like a sentence. Earlier titles: "An Evidence-Bound
   Runtime Contract for LLM Narration of Machine-Learning Pipeline Evidence"
   (2026-09-25) and "Prompts Are Advisory, Verification Is Enforcement",
   dropped because A-L2 = 0/200 contradicted it.
2. **Blind semantic audit (A7), optional now** — A13 gave a validated model
   reading; a human reading would still strengthen the paper. Two authors label
   `benchmark/semantic_audit_A7/audit_sheet.html` independently (204 items,
   one question each, about 1.5 hours), export the CSVs, then run
   `python -X utf8 scripts/make_semantic_audit.py --score a.csv b.csv` and
   update Table 2 (A7 row), RQ1 "What the violations introduce", Threats
   (Construct) and the Conclusion. The sealed key stays local until then.
3. **Budget runs (about 3 USD of DeepSeek; balance was -0.03 USD on 2026-10-02)**
   — both registered as amendment A9 before any call, code ready:
   `reproduce_profile_battery.py --interleave-names --arm tier_a --n 200`
   (naming, interleaved) and `reproduce_hallucination_ablation.py --mode live
   --task synthetic-crowded --retry-audit --n 200` (leakage retry audit, every
   attempt kept); then `scripts/rejudge_rejections.py` and
   `scripts/retry_content.py`, and update RQ5, the rejection audit and Table 2.
4. **Response letter** for the EMSE submission, point by point.
5. **User side** — arXiv; GitHub release -> Zenodo DOI (replace the note in
   `references.bib` entry `replication` and Data Availability).

## What the revision found (all in the manuscript)

- Misfiling = an undescribed field: A-L1-DESCRIBED 1/200 vs 98/200; A-L2 0/200;
  crowded instance 122/200. All 924 flagged responses misfiled, 1 also `target`.
- Author-time enum on data it predates: 145/200 inventions + 132 omissions;
  call-time enum 0/200. Choice list lets 3 through on unchecked fields.
- Profile task: wrong numbers were our suffix-twin naming (0 of 400 under
  natural names); remaining violations = correct class balance in an
  inadmissible slot (fault 13).
- Faults now 13; artifact defects NaN + overflow (falsifier) + empty payload,
  all fixed; falsifier 10^6 cases clean.

---

## 1–3. Free-text battery — DONE (2026-09-24)

All eight leakage arms re-run with the narration recorded, 1,600 responses in
`scripts/runs/2026-09-19_freetext/`, every arm `empty: 0`. Measured with
`scripts/measure_freetext_displacement.py` → `benchmark/freetext_displacement.json`.

Result, now in §12 (Threats, Construct): **displacement not observed.**
Rejecting arms 10.4 % [8.4, 12.7], non-rejecting 9.6 % [7.8, 11.9]; retried
responses 10.1 % vs first-pass 10.5 %. The residue is true bounds, derivations,
and shortened real names; no invented column, no misquoted number. No arm
stripped anything, so the pressure tested is correction, not deletion.

Four repairs on the way, none counted (nothing from them was reported, same rule
as the killed P-CONTRACT run): contract-arm prose emptied twice, Guardrails-arm
prose emptied once (`baseline_guardrails.py` never returned it; empty records in
`superseded_2026-09-24_guardrails_noprose/`), and the first scoring pass
compared signed values (78 % "ungrounded", all artefact).

The run is also a third replication sample, now in the replication paragraph:
A-GR-STOCK 35.0 → 43.5 → 47.5 %, A-STRESS retries 12 → 23 → 32, A-L1 stable,
enforced arms none in all three.

Page count: cut to 15 pages including references on 2026-09-24 (see §5).

---

## 4. Remaining points from the senior review

Writing-level points 1, 2, 3, 4, 5, 8, 10, 11, 13, 14 are **done** (commit
`f8a7ba4`). What is left needs runs or people:

| # | What | Cost | Note |
|---|---|---|---|
| 6 | Wider baseline matrix: prompt-only, schema-only, static enum, dynamic enum, validation-only, validation+retry, full — each with first-pass rate, retries, latency | ~$1 | Most arms exist; the gap is a clean prompt-only and a validation-without-retry arm |
| 7 | Model × mechanism matrix | ~$2–3 | Needs a model that actually fails. `gpt-5.4-mini` and local `qwen2.5:7b` both score 0 unenforced, so they add nothing to a comparison |
| 9 | Adversarial evidence benchmark — near-identical names across fields, nested evidence, duplicates | ~$1 + design | Strongest remaining experiment. The profile task already shows the mechanism: `cat_feature_07` misfiled onto `num_feature_07` by shared suffix. Build the benchmark to provoke that on purpose |
| 12 | Independent human gold labels for the validator | people | Apparatus is ready in `benchmark/blind_review/`, including `rate.html`. Needs two people who are not authors. The Gemini-agent sheets in `pilot_llm/` do not count and must not be reported |
| 16 | More evidence topologies | — | Partly done: RQ6 is a second shape, `--scale-table` measures 10–1000 columns. A nested-evidence shape would close it |

Also open, not from the senior list:

- ~~**τ = 0.005 sensitivity** (Stanford Q3).~~ **Done** (`3025aa8`,
  `scripts/tau_sensitivity.py`): counts identical at every τ from 5e-4 to 0.05;
  in the paper's Measures subsection.
- ~~**The unexplained entity catch** in `P-CONTRACT`.~~ **Re-run done** (`554286d`):
  0 entity violations in 200 more first attempts; the enum held 399/400. The one
  stays unexplained, as the paper says.

## 5. Submission items — the user's side

- **arXiv preprint.** The repository is public and the paper unpublished;
  a dated priority record matters. IEEE allows preprints alongside TSE.
- **GitHub release → Zenodo DOI.** The paper promises it. The integration does
  nothing until a release is actually cut. Once there is a DOI, add it to
  `CITATION.cff` and the Data Availability section.
- **Page count.** Hard limit 15 pages including references, no supplementary
  file. Met: the appendices were folded into §3 and §4, three figures that
  duplicated their tables were dropped, and every section was condensed. Nothing
  measured was removed; any new material must displace something.

---

## Settled — do not reopen

- **Blind audit (#1b / senior #12):** not in the paper. The Gemini sheets are
  filed honestly under `pilot_llm/`.
- **Licence:** stays MIT. Fourteen versions already shipped under it; relicensing
  would not protect them.
- **"main2 was worse":** it was not. The Stanford review labelled main2 described
  main1 (seven faults, rubric pending, attribution intact). Both reviews
  recommended acceptance. The current build supersedes both.
- **0/200:** never presented as a rate of zero. Enforced arms read "none observed
  in 200; below 1.9 % at 95 % confidence", and the guarantee is stated as a
  property of the verifier, not of the sample.

## Instrument faults so far: eleven

Decided: the four repairs made during the free-text battery are named in §12
and not counted, because no number from them was ever reported.
