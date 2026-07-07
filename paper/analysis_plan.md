# Pre-registered analysis plan — Q1 campaign (FabBench measurements)

- **Version:** 1.0 — 2026-07-07 (committed BEFORE any Phase 1+ live run)
- **Scope:** all live measurements feeding the journal manuscript
  (roadmap: `paper/Q1_ROADMAP.md`; requirements R1-R11)
- **Discipline:** every live run is recorded in `paper/ablation_live_*.md`
  with the harness commit hash. Deviations from this plan are allowed but
  must be logged in §8 with a reason BEFORE the deviating run. Selective
  reporting is prohibited: every executed cell is reported.

## 1. Hypotheses (stated before data)

- **H1 (rates vary):** bare-narrator (L1) entity-fabrication rates differ
  across providers on identical evidence; we predict at least a 5x spread
  between the lowest and highest provider on the synthetic task.
- **H2 (contract holds):** the full contract (L3) keeps the user-facing rate
  at 0 on all three channels for every provider and every task
  (upper Wilson bound reported; no provider exceptions).
- **H3 (sweep volatility replicates):** the 6-paraphrase sweep shows a spread
  of at least one order of magnitude (max/min entity-fab rate >= 10x) on at
  least 2 of 3 providers it is run on. Failure to replicate is a reportable
  finding, not a discard.
- **H4 (enforcement dichotomy):** with Tier B disabled and enums present
  (tier_a arm), decode-enforced configurations (OpenAI strict ON; vLLM named
  tool_choice) show 0 entity fabrication; hint-only configurations (same
  model, strict OFF where the provider allows) show > 0 under at least one
  paraphrase. xAI cannot run the hint-only condition (strict is always on) —
  it contributes to the enforced class only.
- **H5 (corrective feedback):** in STRESS arms, the conditional
  repeat-violation rate after one named corrective message is lower than the
  first-attempt rate (one-sided), replicating the June 2026 DeepSeek finding
  (37.5% -> 1.3%) on at least one additional provider.
- **H6 (displacement, exploratory, two-sided):** enforcement pressure may
  push fabrication into the unverified free-text narration field. We compare
  free-text fabrication (LLM-judge + human sample) between L1 and L3/STRESS;
  no directional prediction.

## 2. Provider panel (frozen; changes go to §8)

Per `paper/Q1_ROADMAP.md` §2b (verified 2026-07-07). Subject tier:
deepseek-v4-flash, gpt-5.4-mini, claude-haiku-4-5, gemini-2.5-flash-lite,
mistral-small-latest, grok-4.20-0309-non-reasoning, qwen-flash,
llama-3.1-8b-instant (Groq), Qwen/Qwen2.5-3B-Instruct (local vLLM).
P0 full matrix: deepseek + openai (N=200/cell). P1 breadth: battery + sweep,
N>=100. P2 strong-tier spot-checks (gpt-5.5, claude-opus-4-8,
gemini-3.1-pro-preview), N=100, at most 2 cells each.
Judges (H6): cross-family only — a response is never judged by a model from
the family that produced it.

## 3. Cells and arms

- Arms per task: L1 (bare, open schema), L3 (shipped contract), STRESS
  (contract minus Tier A enums, bare prompt), plus diagnostic tier_a where
  H4 requires it. L2 (strict prompt, unenforced) only on P0 providers.
- Tasks: Phase 1 uses the two June tasks (synthetic, Insurance). Phase 2 adds
  the FabBench task instances (>=4 datasets x >=5 injectors, ~10-12
  instances; the instance list is frozen in this file by amendment BEFORE
  Phase 2 runs — see §8).
- N: 200 per cell on P0; 100 on P1/P2. Smoke runs (N=5) never count as data.

## 4. Metrics (unchanged from the June battery)

Per response reaching the user: entity-fabrication, value-fabrication
(|delta| > VALUE_TOLERANCE = 0.005), critical omission. Wilson 95% CIs and
raw k/N everywhere. Contract arms additionally report Tier B catches
(responses with >=1 rejection; total rejections) and rejection composition
(`rejection_kinds` telemetry). New in this campaign (R6): per-response claim
counts, abstention rates, latency, token usage — logged for every cell via
the harness JSONL run logs.

## 5. Statistical decision rules

- Primary comparisons use two-sided Fisher exact tests at alpha = 0.05:
  L1 vs L3 per provider/task (H2 support), strict ON vs OFF (H4),
  first-attempt vs post-correction violation (H5, one-sided).
- No multiple-comparison correction for the descriptive tables (CIs shown);
  Holm correction applied within each hypothesis family when a hypothesis
  test is quoted in the abstract or conclusions.
- Zero cells are reported with the Wilson upper bound, never as "proven zero".
- An arm that drives the narrator into >= 80% abstention is reported with its
  abstention rate front and center (trivial-zero guard from old Limitation 2).

## 6. Baseline fairness design (Phase 3; R3)

- **Guardrails AI, two arms:** (a) stock loop (JSON validity + reask, no
  faithfulness validator) — measures what the toolkit gives out of the box;
  (b) our Tier B checks wrapped as a custom Guardrails validator — measures
  loop parity (any outcome difference is then attributable to the loop, not
  the validator). We state explicitly in the paper that (b)'s validator is
  ours; the comparison is loops, not validators.
- **NeMo Guardrails:** closest available rail configuration; documented
  config committed to the repo.
- **Outlines / vLLM structured outputs** (local, decode-enforced): same task,
  guided JSON with the evidence enum — the certifiable-decoding comparison
  point. Runs on the open model only (closed APIs cannot host it; that
  asymmetry is itself part of the argument).
- All baselines score on the same three channels + latency + calls/cost.

## 7. Exclusion rules

Only infrastructure failures (HTTP 5xx after retries, provider outage,
harness crash) may exclude a response, and each exclusion is logged in the
run record with its cell and reason. Model refusals, empty tool calls, and
weird-but-parseable outputs are DATA (scored as non-fabricating unless they
violate a channel), never exclusions.

## 8. Amendments log

- **A1 (2026-07-07, BEFORE any Phase 1+ live run):** Phase-2 task-instance
  list FROZEN. Datasets (fetched + SHA256-pinned; see
  `scripts/data/README.md`): insurance/`charges`, heart_cleveland/`chol`,
  telco_churn/`MonthlyCharges`, ames_housing/`SalePrice`. Instances = the 12
  (dataset x injector) cells in
  `scripts/fetch_fabbench_datasets.py::FROZEN_INSTANCES` (every injector on
  exactly 2 datasets) + the frozen June synthetic monotone_log task = 13.
  All 12 real-data instances verified detectable by the shipped detector at
  its thresholds via `fetch_fabbench_datasets.py --verify` (zero API calls)
  before freezing. Injector deviation from the roadmap's tentative list:
  `derived-ratio` replaced by `inverse_target` — a ratio's correlation with
  the target depends on the divisor's variance and cannot be guaranteed to
  cross the detector threshold on arbitrary datasets; the inverse tests the
  same derived-quantity idea deterministically (negative-correlation /
  two-sided path). `binned_target` models the target-encoding CV leak.
- **A2 (2026-07-07, BEFORE any Phase 1+ live run):** Two REAL-WORLD case
  studies added (natural, documented leaks — no injector; `--task case`).
  Expectations pre-stated and verified offline via
  `fetch_fabbench_datasets.py --verify-cases`:
  (i) **bodyfat** (OpenML 560; Johnson, J. Stat. Educ. 4(1), 1996): the
  `Density` feature deterministically generates the target via Siri's 1956
  equation; measured |Spearman| ~ 0.993 >= 0.99 — the shipped detector fires
  on a leak nobody injected (anchor = Density, model = the Siri equation,
  computed r2 = 0.977). Answers the "all your leaks are injected" critique.
  (ii) **sambanis** (Harvard Dataverse doi:10.7910/DVN/KRKWK8, CC0; leak
  cataloged by Kapoor & Narayanan, Patterns 2023 + Neunhoeffer & Sternberg,
  Political Analysis 2019): the documented leak is PROCEDURAL (imputation
  before split) — measured max |feature-target corr| ~ 0.65, zero duplicate
  rows, perfect-match 0.0. Serves as the NEGATIVE CONTROL: the detector must
  stay silent, the contract must abstain (rule 4), and the bare narrator's
  FALSE-POSITIVE fabrication on innocent evidence becomes measurable — a
  direction the injected matrix cannot probe. It also honestly demonstrates
  the detector's scope boundary (procedural leaks are invisible to it;
  paper Limitation 4 territory).
  Survey record: candidates REJECTED with reasons — KDD Cup 2008 (leak is a
  grouping proxy, raw corr << 0.99; redistribution unclear), KDD Cup 1999
  (duplicate channel viable but 74 MB + classification reframing; deferred),
  Kaggle Santander (login-gated, non-redistributable), Bank Marketing
  'duration' (documented but r ~ 0.41, invisible to the detector), battery
  Severson/Geslin (protocol-proxy, needs feature engineering). Case datasets
  are fetched on demand and not committed (bodyfat: no explicit license;
  sambanis: 12 MB); SHA256 pinned in the fetch output.
- **A3 (2026-07-07, BEFORE any Phase 1+ live run) — panel-repair package.**
  Adopted in full from the pre-campaign 3-reviewer panel
  (`paper/review_panel_2026-07-07.md`); every item below supersedes the
  corresponding earlier text.

  **A3.1 — H2 reframed (analytic vs empirical cells).** On the entity and
  value channels, L3/STRESS user-facing zeros are ANALYTIC: Tier B strip
  semantics make them true by construction, so they are evidence of verifier
  correctness (backed by the e2e test suite + an independent re-scorer,
  A3.9), not of model behavior. The preregistered Fisher test of L1 vs L3 is
  WITHDRAWN as vacuous. H2's falsifiable content is re-scoped to: (a) the
  omission channel (flag-only, not strippable per Prop. 2); (b) Tier B catch
  behavior (first-attempt violation rates, retry dynamics); (c) abstention
  rates. The empirically meaningful contrast is L1 vs STRESS first-attempt
  catch rate. All results tables will label analytic cells as such.

  **A3.2 — H5 gains a generic-retry control.** STRESS arms run TWO retry
  variants: (i) named corrective message (existing), (ii) generic rejection
  ("your previous answer was rejected; answer again through the tool")
  carrying no violation details. H5's claim becomes comparative: the named
  variant's conditional repeat rate is lower than the generic variant's
  (one-sided Fisher). Without (ii), only "retry reduces repeats" may be
  claimed. Harness support lands before Phase 1 (code wave).

  **A3.3 — Channel-activation predictions per instance (R2 repair).**
  Predictions frozen now: entity channel = every instance; omission channel
  = all column-anchor instances (contamination + sambanis: UNDEFINED, anchor
  is None — excluded from omission denominators, stated in the paper);
  value channel = primarily the NEW crowded-evidence stressor (A3.4), with
  inverse_target a secondary candidate (a narrator restating the sign it
  reads in prose against the max-abs evidence encoding mismatches the map).
  If the value channel still never fires after A3.4, that outcome is
  reported as a boundary finding and R2 is DOWNGRADED accordingly (decided
  now, so the DoD cannot fail on a hope).

  **A3.4 — New frozen instance: `synthetic_crowded` (value-channel
  stressor).** A synthetic frame (seed 0, spec frozen here) whose evidence
  carries ~10 features with NEAR-IDENTICAL correlations (0.90-0.93 band,
  spaced ~0.003 apart, below/around the candidate threshold except one
  anchor at >= 0.99), so restating any specific value from prose is
  error-prone while entity fabrication is unaffected. Builder lands in the
  code wave; the instance joins the frozen list as #14 BEFORE Phase 2 runs.

  **A3.5 — Sambanis negative-control outcomes registered.** Primary:
  committed-verdict rate (verdict != cannot_determine) — the false-positive
  commitment rate on innocent evidence (contract rule 4 prescribes
  abstention). Secondary: entity-fabrication within its ~10-column evidence
  set; Tier B catches. Omission: undefined (no anchor). The narrator sees
  the COMPUTED r2 of the lstsq fit (honest, non-suspicious) in
  suspicious_metric; the user message is the standard one. The csv-task
  builder will likewise compute r2 from its own y_pred instead of asserting
  1.0 (code wave).

  **A3.6 — Zero-safe ratio rules.** H1 supported iff (largest per-provider
  L1 Wilson LOWER bound) >= 5 x (smallest per-provider L1 Wilson UPPER
  bound), synthetic task, P0/P1 N per §3. H3 supported per provider iff
  (largest variant Wilson lower) >= 10 x (smallest variant Wilson upper);
  sweep providers for H3 are NAMED now: {deepseek, openai, gemini}; "2 of
  3" refers to exactly this set.

  **A3.7 — Precision/power statement.** 0/200 leaves a 1.88% Wilson upper
  bound; 0/100 leaves 3.70%. H4's hint-only ">0 under at least one
  paraphrase" has ~63% per-paraphrase power at a true 1% rate and N=100;
  therefore the two H4-critical tier_a cells (strict-capable provider,
  hint-only mode vs enforced mode) run at N=200. Fisher MDE at N=100 vs
  N=100 (alpha .05, power .8) is roughly 3% vs 15%; provider-difference
  inferential claims are therefore reserved for N=200 cells, N=100 cells
  are descriptive.

  **A3.8 — Holm families fixed now, independent of manuscript placement:**
  F1 = per-provider H2-omission tests; F2 = H4 ON/OFF contrasts; F3 =
  per-provider H5 named-vs-generic contrasts. H5 replication is reported as
  a FRACTION over all STRESS-capable providers run, not as existence.

  **A3.9 — Independent scorer + error semantics.** (a) A separately
  implemented scorer recomputes all three channels from raw JSONL + the
  evidence dict WITHOUT importing the product's corr-map/anchor/tolerance
  helpers; manuscript tables come from it (code wave). (b) Transport
  failures stop scoring as clean data: double-failure responses carry an
  explicit error marker, are excluded per §7 with logged reasons, and
  per-cell error/empty counts appear in every results table.

  **A3.10 — Anchor names stop telegraphing the answer.** Injected columns
  are renamed to dataset-plausible neutral names BEFORE Phase 2:
  exact_copy -> {target}_adj, noisy_proxy -> {target}_est, monotone_log ->
  {target}_idx, inverse_target -> {target}_norm, binned_target ->
  {target}_grp. June's `*_leak` naming is disclosed in Threats to Validity
  as a historical salience confound. A pseudonymized-columns arm (hashed
  names, one dataset x provider cell) is added to measure the
  dataset-familiarity confound directly.

  **A3.11 — STRESS message aligned.** Phase 1+ STRESS arms use the L1 user
  message (no contract mention); the June variant's confound is historical
  and stays disclosed. bodyfat's y_pred moves from the hand-coded Siri
  formula to a least-squares model on all numeric features, with its
  measured r2 reported and the trigger-path caveat kept regardless.

  **A3.12 — Reporting rules.** Headline zero claims are quoted per-cell
  with Wilson upper bounds; pooled 0/N bounds may be reported additionally,
  never as the headline. Both June L1 synthetic measurements (15.0% on
  06-11, 11.5% on 06-12) are reported in the journal version. Abstention is
  operationalized as verdict == cannot_determine from the JSONL logs; a
  >= 80% abstention cell is "unevaluable" for omission claims. The H6 judge
  protocol (prompt file, ordering controls, kappa >= 0.6 judge-human floor
  on a stratified random >= 100 sample) is frozen as a file in the repo
  before any Phase 3 judging.

- **A3 implementation addendum (2026-07-07 evening, code wave; logged BEFORE
  any Phase 1+ live run).** Concrete constants fixed while implementing
  A3.2/A3.4/A3.9/A3.10/A3.11 in the harness:

  **Sampling pin.** All live cells send `temperature = 1.0` (the shared
  cross-provider setting; `top_p` is never sent). If a provider rejects the
  parameter (reasoning endpoints), the pin is dropped for the retry and the
  record's `sampling.temperature` is logged as null — per-record, so mixed
  cells are visible. The pin is a nuisance-variance control for H1's
  cross-provider comparison, not a claim about optimal decoding; `--temperature`
  can override per run and the value used is recorded in every JSONL record.

  **Instance #14 frozen parameters (A3.4).** `synthetic_crowded`: n = 1200,
  seed 0; anchor `sensor_ref` at exactly 0.995; nine crowd features
  `sensor_00..sensor_08` at exactly 0.900, 0.903, ..., 0.924 (0.003 grid).
  Sample Pearson correlations are constructed exactly (in-sample Gram-Schmidt),
  so the detector reports the grid values verbatim; Spearman of the Gaussian
  mix stays strictly below Pearson, so max(pearson, spearman) = the grid.
  `y_pred = y + N(0, 0.001*sd(y))`; suspicious r2 COMPUTED from those
  predictions (A3.5 discipline). Neutral names per A3.10. Verified by
  `fetch_fabbench_datasets.py --verify` (grid to 1e-6) and by tests.

  **Independent scorer operationalization (A3.9a).** `scripts/independent_scorer.py`
  restates tolerance 0.005 and the channel rules without importing product
  or harness helpers. Clarified operationalizations, frozen now: an
  off-evidence CLAIM column is an entity violation (the value channel is
  reserved for wrong numbers on real columns); omission requires a
  substantive committed answer (>= 1 citation or claim AND verdict neither
  empty nor cannot_determine); the contract's user-facing `omitted` flag,
  where present in a record, is read as data (it is the product's post-strip
  output; Prop. 2), while open arms are recomputed from raw fields.
  Manuscript tables are generated only by `scripts/make_tables.py` from
  committed JSONL logs plus the harness's `evidence_<hash>.json` dumps, and
  each table carries the git commit + scorer cross-check disagreement count.

  **Pseudonymized-columns arm (A3.10).** `--pseudonymize` (csv task only)
  renames every column (target and anchor included) to
  `col_<sha256(name)[:8]>` before evidence building; task label gains a
  `/pseud` suffix (distinct cells); the name mapping is printed to stderr
  for the audit trail. One dataset x provider cell in Phase 2 as registered.

  **Error-marker scope note (A3.9b).** The 2-attempt transport policy and
  error markers now cover ALL live paths including the paraphrase sweep;
  sweep scoring excludes error records the same way the battery does.

