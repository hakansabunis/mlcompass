# Live measurement battery — stress arm, paraphrase sweep, real-data task

- **Date:** 2026-06-12
- **Provider / model:** DeepSeek OpenAI-compatible endpoint, `deepseek-chat`, provider-default sampling
- **Harness:** `scripts/reproduce_hallucination_ablation.py` @ commit 17a2766 (stress arm + sweep + csv task)
- **Value tolerance:** 0.005. Layer 3 / STRESS route through the shipped `investigate_leakage_bound`.

## Run A — synthetic task, 4 arms, N=200/arm

Evidence columns: 10, anchor `log_target_v2`, seed 0.

| Arm | Entity-fab | 95% CI | k/N | Value-fab | Omission |
|---|---|---|---|---|---|
| L1 bare prompt | 15.0%→**11.5%** | [7.79, 20.61→16.66] | 23/200 | 0/200 | 0/200 |
| L2 + strict prompt | 0.0% | [0.00, 1.88] | 0/200 | 0/200 | 0/200 |
| L3 + full contract | 0.0% | [0.00, 1.88] | 0/200 | 0/200 | 0/200 |
| STRESS (bare + Tier B only) | **0.0% user-facing** | [0.00, 1.88] | 0/200 | 0/200 | 0/200 |

(Note: 11.5% is this run; the 2026-06-11 run of the same L1 configuration measured
15.0% — within sampling noise, ~1σ.)

Tier B catches: L1/L2/L3 = 0; **STRESS = 4 responses with ≥1 catch, 4 total catches**
(each corrected on the first retry; zero repeat violations; zero residual strips).

## Run B — bare-prompt paraphrase sweep, 6 variants × N=100 (synthetic)

Six rule-free paraphrases of the same instruction; same model, same evidence,
same open tool schema; only wording varies.

| Variant | Entity-fab | 95% CI | k/N |
|---|---|---|---|
| terse | 1.0% | [0.18, 5.45] | 1/100 |
| baseline | 7.0% | [3.43, 13.75] | 7/100 |
| cautious | 45.0% | [35.61, 54.76] | 45/100 |
| expert | 90.0% | [82.56, 94.48] | 90/100 |
| helpful | 93.0% | [86.25, 96.57] | 93/100 |
| mechanical | 100.0% | [96.30, 100.00] | 100/100 |

**Range: 1% → 100%.** The bare-prompt fabrication rate of the same model on the
same evidence is essentially unpredictable from wording.

## Run C — real-data task (Insurance Charges, injected log leak), 4 arms, N=200/arm

Evidence columns: 7, anchor `log_charges_leak`. Real third-party dataset
(public Insurance Charges CSV); leak + near-perfect predictions injected; the
shipped `detect_leakage` produced the evidence.

| Arm | Entity-fab | 95% CI | k/N |
|---|---|---|---|
| L1 bare prompt | 43.5% | [36.82, 50.43] | 87/200 |
| L2 + strict prompt | 0.0% | [0.00, 1.88] | 0/200 |
| L3 + full contract | 0.0% | [0.00, 1.88] | 0/200 |
| STRESS (bare + Tier B only) | **0.0% user-facing** | [0.00, 1.88] | 0/200 |

Value-fab and omission: 0/200 in every arm.

**Tier B catches (STRESS): 75/200 responses with ≥1 catch, 76 total catches,
user-facing violations 0/200.** Only one response violated again after the
corrective retry message: conditional repeat-violation rate 1/75 ≈ 1.3% vs the
37.5% first-attempt rate — a ~28× drop. No response reached the strip fallback;
all became faithful via retry alone. No omission flags.

## Retry-model check (pre-registered in conversation before Run C)

Under an iid per-attempt violation model with q = first-attempt rate (0.375),
expected total catches ≈ N·(q + q² + q³) ≈ 114. Measured: 76. The iid model is
rejected in the informative direction: the corrective message collapses the
repeat-violation probability (sub-iid), so the contract's expected overhead is
E[calls] ≈ 1 + q·(1 + 0.013) ≈ 1.38 even in the stress configuration, and ≈ 1.0
in the production configuration (strict prompt, zero violations observed).

## Honest confound note

The STRESS arm's user message is the shipped contract's message, which mentions
a "strict contract" (its system prompt is bare); L1's user message does not.
First-attempt propensities are therefore not exactly comparable between L1 and
STRESS (synthetic: 11.5% vs 2%; insurance: 43.5% vs 37.5%). Catches are reported
as demonstrations of the verifier firing, not as estimates of the bare rate.

## Run D (second model) — not executed

`deepseek-reasoner` arm was offered but not run; the paper makes no
second-model claim.
