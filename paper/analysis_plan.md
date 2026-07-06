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
