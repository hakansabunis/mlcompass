# 12-page version: outline and evidence map

ARS `academic-paper`, outline-only mode (Phase 0 → 2). Source: `../tse_latex/main.tex`
(the full paper, which becomes the online extended version). Draft: `main.tex` in
this directory.

## State after the review round (2026-10-08)

A five-reviewer ARS panel (reports in `paper/reviews/short_v1/`, kept local) asked for
major revision with no new data. Applied: Table II lists every registered item with its
status; the registered 2-point equivalence band and 5-point materiality bar are applied
(X3 not material, X4 inside the band); the losing-condition sentences LC-1, LC-3 and
LC-5 are printed; A11's history (A3 R1 and R2b stopped, the interim look) is disclosed;
the strict flag went to the provider's standard endpoint and may not have applied; the
judge validation could not test the B label; the closest work (FAX, Huang and Deng,
ProvenanceGuard, VERITYGATE) and error typologies (FRANK, Thomson and Reiter, slot
aligners) are cited; terms are defined; tables are no longer shrunk; the DOI of the
replication package and the extended version is printed. The same factual corrections
went into the extended version.

Build: 12 pages including references; text ends at 83 % of page 10 (9.8 pages),
references end at 38 % of page 12. 8,247 words of prose, 7 tables, Fig. 1, Algorithm 1,
57 references. Checks: `measure.py` (pages, fonts, abstract, the pronoun rule) and
`check_numbers.py` (every number is in the extended version).

## Phase 0: Paper Configuration Record

| Parameter | Value |
|---|---|
| Topic | Runtime contracts bound to a deterministic tool's output, for LLM narrators of that output |
| Research questions | RQ1 how an unconstrained narrator fails and what removes the failure; RQ2 what the contract guarantees and costs, against the alternatives a practitioner would use; RQ3 whether contract and failure carry over to a second evidence shape |
| Paper type | IMRaD, empirical software engineering (controlled experiment) |
| Discipline | Software engineering / trustworthy LLM systems |
| Target | One version for every venue: IEEE TSE (free up to 12 pages including references and biographies), conference format 10 + 2, and EMSE (no limit) |
| Citation format | IEEE (IEEEtran) |
| Output | LaTeX, IEEEtran journal, two columns |
| Language | English |
| Abstract | 150 to 200 words, five parts (problem; prevailing approaches and their limits; proposal; results; contributions); no dashes, hyphens or apostrophes |
| Length | 10 pages of text (title to Data Availability) + 2 pages of references; about 6,300 words of prose, 5 tables, Fig. 1 |
| Existing materials | The full paper (26 pages, 18,355 prose words, 19 tables, 60 references), all run records and scripts |
| Co-authors | H. Sabuniş (corresponding), Y. Ünlü, M. K. Özdemir; Istanbul Medipol University |
| Funding | none |
| Operational mode | outline-only, then condensation by hand from this outline |

Confirmed by the user on 2026-10-06 ("12 sayfalık okey ... geri kalanı online").

## Phase 1: Literature

Skipped: the sources are the full paper's 60 verified references. The 12-page version
keeps 47 of them and adds the extended version (list below); the other 13 stay in the
extended version.

## Phase 2: Outline (word budget = prose, floats excluded)

| § | Section | Words | Floats | Was (words) |
|---|---|---|---|---|
| – | Abstract | 190 | | 250 |
| I | Introduction | 850 | | 1,619 |
| II | Background and Related Work | 550 | Table I taxonomy | 896 |
| III | The Evidence-Bound Contract | 750 | Fig. 1 | 2,323 (§III + §IV) |
| IV | Study Design | 700 | Table II registered predictions | 2,281 |
| V | RQ1: How the Unconstrained Narrator Fails | 850 | Table III kinds | 1,847 + 391 (old RQ4) |
| VI | RQ2: What Enforcement Guarantees and Costs | 1,100 | Table IV arms | 2,900 |
| VII | RQ3: A Second Evidence Shape | 450 | Table V profile | 1,263 |
| VIII | Discussion | 400 | | 887 |
| IX | Threats to Validity | 450 | | 1,054 |
| X | Conclusion | 170 | | 387 |
| – | Acknowledgments, AI declaration, Data Availability | 330 | | 431 |
| | **Total** | **≈ 6,600** | 5 tables, 1 figure | 18,355 |

Section purposes and transitions:

- **I** problem (narrators of ML tools), the leakage example, invention vs misfiling, the three
  places defenses act and their limits, Definition 1, the contract in one paragraph, three
  RQs, headline numbers, four contributions, scope. → II places the contract among
  existing enforcement.
- **II** closed-world faithfulness and data-to-text; classical enforcement (DbC, edit
  automata); decode-time and schema methods; the closest validate-and-reask work.
  → III states the contract precisely.
- **III** E, per-field admissibility, Definition 2 (C1–C3), misfiled / unlisted / invented,
  Tier A / Tier B, deletion guarantee and repair asymmetry in prose, Fig. 1, how the
  implementation was tested, the generic layer, what the invariant does not cover.
  → IV says how it was measured.
- **IV** pre-registration and Table II, subjects, models and sampling, arms, measures and
  kinds, instrument faults in two sentences. → V–VII answer the RQs.
- **V** floor rate, composition (1,807 / 1,829), twelve frozen instances, real data, Table III,
  semantic audit (A13), what removes the failure (description, prompt, T = 0, wording).
- **VI** Table IV, the toolkit comparison, binding at call time and A14, the registered test
  A11, cost, the rejection audit, why Tier B is still needed, providers without failures.
- **VII** profile task, wrong numbers and naming (A9.1), how the tiers separate.
- **VIII** four design lessons, simpler designs, where the contract transposes.
- **IX** construct, internal, external, conclusion validity.

## Evidence map: what stays, where

| Claim / number | § |
|---|---|
| Floor 42.0 % [35.4, 48.9]; four runs 42.0–49.0, χ² p = 0.34 | V |
| 1,807 of 1,829 flagged misfile, 13 invented (11 the word `target`) | I, V, X |
| 12 frozen instances: 817 / 1,888; R2a held at 1.5 % | IV (Table II), V |
| Real data 21.5 % / 20.5 %, always `r2` | V |
| Kinds: 1,831 = 1,772 misplaced + 57 extrinsic + 1 + 1; 145 of 152 on the stale enum | V, Table III |
| A13 judge 55/55, 73/79 (64/65 corrected), `r2` false relation 40/40 [91, 100]; A10, A12 failed | V, Table II |
| Description 49.0 → 0.5 %; rules 7.0 %; shipped 0/200; T = 0 2.0 %; crowded 61.0 % | V |
| Wording sweep 9.0–72.0 %; pairs not carried 0 to 45 per hundred | V |
| Guardrails stock 43.5 vs choices 0 (same day); 3 vs 0 on crowded | VI |
| Typed schema 4.0–7.0 %; call-time enum 0 | VI |
| Stale enum 145/200 invented (72.5 %), 139 omit anchor; A14 143/2/0/3/0, trend p < 10⁻⁷⁶, P2 p = 0.25 | VI |
| A11 10/600 vs 0/600, p = 0.002, [0.7, 3.0]; 29.4 correct claims; 5.47 vs 5.44 s | I, VI, X |
| 17 µs per response, 15 µs binding; 3.3 vs 2.8 s; nothing stripped | VI |
| 24 + 17 rejections confirmed independently; 170 claims before and after | VI |
| gpt-5.4-mini 0 in 200; qwen 2/20 | VI |
| Profile: 24 vs 10 columns, 13 vs 1 statistics, 196 vs 10 quantities; 9/200 wrong numbers, 11 of 12 sibling values; A9.1 7 vs 1, p = 0.068; tiers 20→1, 8→5, 9 vs 11 | VII |
| 98.7 % of 1,831 use a name in E; four of five judges failed | VIII |
| Free-text 8.8 vs 12.6 %; 512 of 2,400 calls lost; power figures | IX |

## Moved to the extended version (nothing deleted)

Table of all registered items with timestamps and deviations (old Table 1, A1); arms
definition table; statistic alias table; four-questions table; inventory table; terms
table; delivery table; retry table; wording-sweep table; conditions table; policy
sensitivity table; corrections table and the seventeen faults in detail; rejection
table (A2); Algorithm 1; the repair-asymmetry observation and proof sketch; the
tolerance and integer-claim analysis; abstention and verdict counts; drift between runs;
deletion-without-retry analysis; the class-balance re-scoring; the failed judges in
detail; the free-text scan in detail; the downstream benchmark.

## References kept (47 of 60, plus the extended version)

react, toolformer, mcp, ji, huang, maynez2020faithfulness, leakage, ydata, mlflow,
outlines, gcd, scholak2021picard, poesia2022synchromesh, structured, liu2023instructor,
guardrailsai, nemo, reiter1978closed, dusek2020e2e, wiseman2017challenges,
dhingra2019parent, meyer1992dbc, ligatti2005edit, schneider2000enforceable,
signe2026chyd, cruz2026atlasrtc, geng2025jsonschemabench, le2026schemakey,
ahn2026harness, zhang2025rvllm, winston2026solver, du2026ledgermind, chen2025evibound,
greshake2023injection, ralph2020standards, penrose1985bodyfat, muchlinski2016civilwar,
lantz2013mlr, janosi1988heart, ibm2019telco, decock2011ames, sclar2024formatspread,
yang2022leakage, truong2025leakagedetector, wang2026agenttraces, ng2026runtimecontract,
replication, extended (new).

Dropped here, kept in the extended version: tensorflow, autosklearn, codex, swebench,
lmql, guidance, xu2025openworld, luo2026removal, karnatak2026ava, falcone2012verify,
agrawal2023mgd, li2026orderbench, zhang2025cda. `check_numbers.py` prints both lists.
