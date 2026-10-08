# Outline and evidence map (V6, Q1 flow)

ARS `academic-paper`, outline-only mode. Source draft: `main.tex` (V5, commit 0e47720).
Every number comes from `../tse_latex/main.tex` (the extended version) and is checked
by `check_numbers.py`. Earlier outlines are in git history (V4.0: 64f3894).

## Phase 0: Paper Configuration Record

| Parameter | Value |
|---|---|
| Topic | Faithful narration of deterministic tool output by LLMs, enforced by a runtime contract bound to that output |
| Thesis | When the citable entities and quantities of a narration are fixed at call time, admissibility can be defined per answer field and enforced by a membership check; measured against that definition, a failing narrator mostly misfiles rather than invents |
| Framing | An idea and its evaluation. mlcompass is the testbed, not the subject; no audience-specific framing. The word "engineer" does not appear (the affiliation "School of Engineering and Natural Sciences" is a proper name and stays) |
| Paper type | IMRaD, empirical software engineering, controlled experiment with registered tests |
| Target | Q1 SE journal (TSE, EMSE) and 10+2 conference formats; 10 to 12 pages including references |
| Citation format | IEEE |
| Abstract | 150 to 200 words in five parts: problem; prevailing approaches and their limits; proposal; results; contributions. No dashes, hyphens, or apostrophes |
| Constraints | No pronoun "it"; same data and numbers; three RQs |
| Mode | outline-only, then drafting by hand |

Confirmed by the user on 2026-10-08 (five-part abstract, no "engineer", Q1 flow, ARS
outline and abstract skills).

## Phase 1: Literature

Skipped: the 51 verified references of V5 are kept; no source is added or removed.

## Phase 2: Outline

Argument flow: context → problem → why the error kind matters → gap → idea →
evaluation → findings → implications → limits.

| § | Section | Words | Floats | Purpose |
|---|---|---|---|---|
| – | Abstract | 190 | | Five parts, as configured |
| I | Introduction | 950 | | Context, problem, gap, idea, RQs, findings, contributions, scope |
| II | Background and Related Work | 650 | Table I | Typologies, where constraints act, closest work |
| III | Evidence-Bound Narration | 1,050 | Fig. 1, Alg. 1 | Definitions, contract, guarantee and its limits, implementation |
| IV | Evaluation Design | 950 | Table II | Testbed, tasks, narrators, configurations, measures, registration |
| V | RQ1: How narrators fail | 850 | Table III | Composition, kinds, semantic reading, what removes the failure |
| VI | RQ2: Enforcement and its alternatives | 1,250 | Table IV | Mechanisms, call-time binding, registered verifier test, cost |
| VII | RQ3: A second evidence shape | 450 | Table V | Profile task, misfiled values, tier separation |
| VIII | Implications | 450 | | Four design principles for tool narrators; alternatives; transfer |
| IX | Threats to Validity | 450 | | Construct, internal, external, conclusion |
| X | Conclusion | 200 | | Thesis restated with the evidence |
| | **Total** | **≈ 7,300** | 5 tables, 1 figure, 1 algorithm | |

### I. Introduction (≈ 950)
- **I.1 Context.** LLMs increasingly sit between deterministic tools and the people or
  programs that consume their results: an agent calls a tool, reads the structured
  output, and narrates the result [react, toolformer, mcp]. The narration inherits
  the authority of the tool.
- **I.2 Problem, with one concrete case.** A leakage detector reports $R^2=1.0$; the
  narrator cites `r2` as a leaking column in 42 % of answers on one model. `r2` is the
  metric's name, carried by the evidence. The structured field is consumed by
  programs and displayed as verified [ji, huang].
- **I.3 Why the kind of error matters.** Invention (extrinsic) vs misfiling (intrinsic)
  [ji, maynez]; different remedies; one rate hides which applies.
- **I.4 Gap.** Constrained decoding needs logits; provider schemas are unobservable;
  validate-and-reask leaves admissibility to the author and stock installs check
  structure only (Table I).
- **I.5 Idea.** Evidence-closed narration (Definition 1); field-indexed admissibility
  derived at call time; two tiers; the mechanism is not new, the definition and what
  the definition reveals are.
- **I.6 Evaluation and findings.** Three RQs; mlcompass as an open testbed; 19,132 +
  3,200 responses; three findings (misfiling dominates; author-time lists turn
  misfiling into invention; the check is cheap, ties enumerations on names, removes a
  residue in the registered test).
- **I.7 Contributions** (definition; failure account; comparison) and **scope** (one
  failing model; no general claim).
- *Transition:* the gap is positioned against prior work.

### II. Background and Related Work (≈ 650)
- Faithfulness to a finite record: data-to-text metrics [dusek, wiseman, dhingra];
  typologies [pagnoni, thomson]; generation-time slot checks [wen, juraska]; what
  differs for a commercial API. Closed world [reiter]; DbC [meyer]; edit automata
  [ligatti]; enforceability [schneider].
- Where constraints act (Table I): decode time [gcd, picard, synchromesh, chyd,
  atlas-rtc]; schemas [geng, le]; validate-and-reask [instructor, guardrails, nemo];
  policy contracts [ahn, rvllm, winston]; ledger [ledgermind]; EviBound.
- Narration under a check: FAX, Huang and Deng, ProvenanceGuard, VERITYGATE; none
  indexes admissibility by field or measures misfiling against invention.
- *Transition:* the contract is defined.

### III. Evidence-Bound Narration (≈ 1,050)
- Evidence in the leakage task; Definition 1; why a flat membership test fails (`r2`);
  entity fields, $A_{E,col}$, $V_E$, anchor, commit; Definition 2 (C1–C3); channels;
  misfiled / unlisted / invented.
- Two tiers; Fig. 1; Algorithm 1; the deletion guarantee for (C1)–(C2), reported
  without interval; (C3) flagged (repair asymmetry); anchor re-check after stripping.
- Implementation: task-agnostic layer, forty-line specification, $O(|r|)$, pair-keyed
  value table, tests (36 mutation, 10 fault injection, $10^6$ property cases built after
  the runs, two defects, NaN fix date).
- What the contract does not cover: verdict, free text, trusted producer, other
  surfaces, injection through column names [greshake, zhang-cda].
- *Transition:* how the idea is evaluated.

### IV. Evaluation Design (≈ 950)
- Testbed: mlcompass, open source, shipped leakage path; controlled experiment
  [ralph].
- Tasks (reference, crowded, bodyfat, sambanis, twelve frozen); narrators and
  sampling; configurations (floor, shipped prompt, field description verbatim,
  Guardrails trio, strong pair, strict flag caveat); measures (violations, kinds,
  Wilson, tolerance, two scorers, seventeen faults); registration (Table II, band and
  bar, structural zeros, status of remaining items, three deviations).
- *Transition:* results by RQ.

### V–VII. Results
- **RQ1:** floor 42.0 %; four tasks 1,012; real data 21.5 / 20.5 %; twelve frozen
  817 / 1,888, prediction held; 1,807 / 1,829 and 13; Table III kinds (1,831 =
  1,772 + 57 + 1 + 1; stale enum 145 / 152); A13 judge (55/55, 73/79, 64/65; 40/40,
  contrary to prediction; no B items); remedies (0.5, 0, 7.0, 2.0, 61.0; other models;
  wording 9–72 %).
- **RQ2:** Table IV; LC-1 parity on names; LC-5 (check and binding, not loop);
  Guardrails 43.5 vs 0, crowded 3 vs 0; typed schema 4.0–7.0, not material; stale list
  145 / 200, A14 143 / 2 / 0 / 3 / 0, trend carried by the empty list, 8 of 10
  indistinguishable; strong arms 0 / 200, pooled [−0.5, 0.5]; A3 stopped (93, 91, 50);
  A11 10 / 600 vs 0 / 600, interim look, slips, content and latency; cost (17 µs,
  15 µs, 3.3 vs 2.8 s, retries, nothing stripped, content kept, shipped prompt a
  quarter); rejections (24 + 17); why Tier B.
- **RQ3:** profile shape (24 / 13 / 196); misfiled values 9 / 200, 11 of 12; natural
  names 0 / 400; A9.1 7 vs 1; tiers separate (20→1, 8→5, 9 vs 11); X2 void (LC-3);
  class balance 26→10, 17→10.

### VIII. Implications (≈ 450)
- Four principles for systems that narrate tool output: index admissibility by field
  (98.7 %); bind at call time; describe fields but keep the check (fix effects are
  configuration-specific, 20–61 %); report violations apart from what they introduce
  [pagnoni, thomson]; judges are not a substitute (four of five failed; two cents vs
  microseconds).
- Alternatives not run (host-rendered numbers, derived verdict [huang-deng],
  relational schema); retry vs strip (6–16 %, 41.5–59 %); transfer conditions.

### IX–X. Threats and Conclusion
- Construct (consistency with E; S4 rejected truths; policy re-scoring; judge; free
  text 8.8 vs 12.6 %); internal (blocks, rolling alias); external (one provider;
  four of sixteen tasks; 512 lost calls; default narrator not measured; enum size);
  conclusion validity (power 0.8 / 0.21 / 0.89).
- Conclusion: the thesis with its evidence and its limit.

## Evidence map

Unchanged from V5: every number listed above appears in `../tse_latex/main.tex`;
sources per section as listed. No new source.

## Changes from V5

1. Every mention of an "engineer" is replaced by the reader, the user, or a consuming
   program; the opening states the general setting before the case.
2. The introduction follows context → problem → gap → idea → evaluation; the
   Discussion becomes "Implications" for systems that narrate tool output.
3. The abstract is rewritten with the ARS abstract mode in the five configured parts.
