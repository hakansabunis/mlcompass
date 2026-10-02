# Where this stopped — 2026-10-02

**V2.3: DONE.** A full readability pass (the abstract restructured as problem /
prior approaches / proposal / results / contributions; no ambiguous "it";
shorter sentences; American spelling) and the V2.2 EMSE review addressed:
kinds of violation replace the "unsupported" layer, every stored rejection is
re-judged, a same-day naming comparison, fault 16. EMSE build: 43 pages, no
warnings, abstract 246 words. Analysis plan amendments A6-A8.

## Next

1. **Title** — changed 2026-10-01 to "Misfiling, Not Invention: How an LLM
   Narrator of Machine-Learning Pipeline Evidence Fails, and What an
   Evidence-Bound Contract Checks" in the TSE and EMSE manuscripts, README
   and CITATION.cff, after the V2.1 review read the paper as an empirical
   diagnosis rather than a mechanism. Earlier titles: "An Evidence-Bound
   Runtime Contract for LLM Narration of Machine-Learning Pipeline Evidence"
   (2026-09-25) and "Prompts Are Advisory, Verification Is Enforcement",
   dropped because A-L2 = 0/200 contradicted it.
2. **Blind semantic audit (A7)** — two authors label
   `benchmark/semantic_audit_A7/audit_sheet.html` independently (204 items,
   one question each, about 1.5 hours), export the CSVs, then run
   `python -X utf8 scripts/make_semantic_audit.py --score a.csv b.csv` and
   update Table 2 (A7 row), RQ1 "What the violations introduce", Threats
   (Construct) and the Conclusion. The sealed key stays local until then.
3. **Budget runs (about $5 of DeepSeek)** — register each in the analysis
   plan first: (a) the two naming schemes interleaved on one harness
   (`P-L1`, `P-TIER-A-ONLY`); (b) a leakage retry battery that keeps every
   attempt (`A-STRESS` on the crowded instance, `A-CONTRACT`); (c) complete
   A3 R1.
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
