# Where this stopped — 2026-09-25

**IN PROGRESS: revision after the ARS 5-reviewer panel (Major Revision).**
Reviews and the editorial decision are in `paper/reviews/2026-09-24_ars_panel/`
(not committed — the repo is public; they are local only). Roadmap: P0-1..P0-16
in `editorial_decision.md`.

## 0. First thing tomorrow: finish the last battery (~825 calls, ~1 h)

```bash
source scripts/load_keys.sh
python -X utf8 -u scripts/reproduce_hallucination_ablation.py --mode live   --provider deepseek --n 200 --resume --task synthetic-crowded --only-baselines   --log-dir scripts/runs/2026-09-24_revision
python -X utf8 scripts/analyze_revision_runs.py --json benchmark/revision_runs.json
```

Remaining cells on the crowded instance: guardrails_stock (194/200),
guardrails_choices, guardrails_tierb, static_schema_noenum, **static_schema =
A-STATIC-ENUM-STALE, the arm P0-3 needs** (author-time list covers 0/10 of the
crowded columns). Balance: ~$5 was added 2026-09-24.

## Results so far (all N=200, deepseek-chat, re-scored by analyze_revision_runs.py)

| arm | result |
|---|---|
| A-L1 (09-24) | 98/200, all misfiled r2 |
| **A-L2** (shipped rule-bearing prompt, no enforcement) | **0/200** — as the plan predicted in §2.1 |
| **A-L1-DESCRIBED** (field description only) | **1/200** — DA C2 confirmed: misfiling = under-specified field |
| A-L2-DESCRIBED | 0/200 |
| A-L3-SHIPPED | 0/200 (1 malformed retry — new abort path exercised live) |
| A-STRESS / -GENERIC | 0 user-facing; 19 / 29 responses retried |
| A-STRICT-ENUM with --strict | 0/200 (H9) |
| crowded A-L1 / A-GR-STOCK | 122/200 / 110/194, all r2 |
| crowded A-CONTRACT / A-STRESS / live enum | 0 / 0 (25 retried) / 0 |
| RQ5 profile, natural names | 0 wrong numbers in 400; remaining violations = correct class balance in an inadmissible slot (fault 13) |

Drift (`scripts/drift_tests.py`): A-L1 stable (p=0.34, pooled 43.6%);
A-GR-STOCK heterogeneous (p=0.035, pooled 42.0%); A-STRESS retries 12/23/32 (p=0.006).

## Manuscript state (`paper/tse_latex/main.tex`, commits 9ecc2e6 onward)

Rewritten: related work (Table I = locus x binding time), theory (Def. 2 over
JSON paths, property renamed "referential and value soundness"), artifact
(abort path, threat model, three narration surfaces), study design (+ Table
`tab:registered`, registered items and status), RQ4 (specification, `target`
= 18 of 20 "inventions", no co-variation), RQ5 (ex-RQ6, natural names),
corrections (one table, 13 faults), RQ5-downstream cut.

**Still to write, after the last battery:** abstract, intro (RQs, contributions,
reframe: every observed failure disappears under some reasonable
specification change — rule prompt, field description, distinct names — and
reappears under another; the contract's value is a guarantee independent of
all of them), RQ1 (misfiling = under-specified field; description control;
crowded instance), RQ2/3 table (add A-L2, A-L3-SHIPPED, strict enum, crowded +
stale enum; drop "<1.9%" as evidence for verifier arms, keep per-cell bound
per A3.12 labelled as describing the sample), Discussion (+ one paragraph on
the cut downstream benchmark and where it lives), Threats, Conclusion, the
remaining factual items of editorial_decision.md Part 3, then compile and fit
15 pages. The build currently has dangling refs (sec:rq5-rubric, tab:rubric).

Title question for the user: keep "Prompts Are Advisory..."? A-L2 = 0/200
means a reviewer will read the title as contradicted; argument is about
certification, not effectiveness.

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
