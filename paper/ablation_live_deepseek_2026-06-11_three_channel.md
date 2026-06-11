# Live three-channel ablation record — Table I certification run (v2)

- **Date:** 2026-06-11
- **Command:** `python scripts/reproduce_hallucination_ablation.py --mode live --provider deepseek --n 200`
- **Provider / model:** DeepSeek OpenAI-compatible endpoint, `deepseek-chat`
- **Sampling:** provider-default settings (no temperature override)
- **Synthetic frame seed:** 0 — evidence columns: 10, anchor (top candidate): `log_target_v2`
- **Value tolerance:** 0.005 (two-decimal rounding passes)
- **Layer 3 path:** shipped `investigate_leakage_bound` (Tier A runtime enums on
  `columns_referenced` and `claims[].column` + Tier B deterministic verification of
  entity soundness, claim values, and completeness).

## ⚠ Relation to the 2026-06-10 run

The 2026-06-10 run (`ablation_live_deepseek_2026-06-10.md`) measured the entity
channel only, under a bare prompt that did **not** request structured claims.
That variant produced **56.5%** (113/200) entity fabrication. This run's bare
prompt adds one sentence requesting structured `{column, statistic, value}`
claims — and the entity rate fell to **15.0%** (30/200). Same model, same task,
same seed: a near-fourfold swing from a seemingly innocuous prompt change.
Both records are kept deliberately: together they are direct evidence that
prompt-level fabrication rates are volatile and cannot be relied upon, which is
the motivating argument for the structural contract.

## Raw output (verbatim)

```
## Three-channel contract-violation rates (live mode, N = 200)

### Entity-fab rate
| Layer                   | Rate  | Wilson 95% CI    | k / N      |
| ----------------------- | :---: | :--------------: | :--------: |
| L1 bare prompt          | 15.0% | [10.71, 20.61] |  30 / 200  |
| L1+2 strict prompt      |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| L1+2+3 evidence-bound   |  0.0% | [ 0.00,  1.88] |   0 / 200  |

### Value-fab rate
| Layer                   | Rate  | Wilson 95% CI    | k / N      |
| ----------------------- | :---: | :--------------: | :--------: |
| L1 bare prompt          |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| L1+2 strict prompt      |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| L1+2+3 evidence-bound   |  0.0% | [ 0.00,  1.88] |   0 / 200  |

### Omission rate
| Layer                   | Rate  | Wilson 95% CI    | k / N      |
| ----------------------- | :---: | :--------------: | :--------: |
| L1 bare prompt          |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| L1+2 strict prompt      |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| L1+2+3 evidence-bound   |  0.0% | [ 0.00,  1.88] |   0 / 200  |
```

## Reading

- **Entity channel is the live failure mode**: 15.0% under this bare prompt
  (56.5% under the 06-10 variant). The strict prompt and the full contract both
  eliminated every observed violation (0/200).
- **Value channel**: no misquotes observed in any configuration (upper Wilson
  bound 1.88%). When asked to copy correlation values into structured claims,
  this model copied them accurately on this task. The contract closes the
  channel by construction regardless; the risk may grow with longer contexts
  or derived statistics.
- **Omission channel**: no critical omissions observed; the anchor is the most
  salient item in the evidence. The contract retries on omission and flags what
  survives the budget (an omission cannot be stripped).
- Composition of L1 entity violations and per-response claim counts were not
  logged in this run.
