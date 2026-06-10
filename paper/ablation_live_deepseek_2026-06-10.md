# Live ablation record — Table I certification run

- **Date:** 2026-06-10
- **Command:** `python scripts/reproduce_hallucination_ablation.py --mode live --provider deepseek --n 200`
- **Provider / model:** DeepSeek OpenAI-compatible endpoint, `deepseek-chat`
- **Sampling:** provider-default settings (no temperature override)
- **Synthetic frame seed:** 0 (default) — evidence columns: 10
- **Layer 3 path:** shipped `mlcompass.agents.leakage_investigator.investigate_leakage_bound`
  (Tier A runtime enum + Tier B deterministic validation). Note: this endpoint does
  not strictly enforce schema enums during decoding, so Tier B is the operative
  enforcement in this configuration.

## Raw output (verbatim)

```
## Phantom-column fabrication rate (live mode, N = 200)

| Layer                            | Rate  | Wilson 95% CI    | k / N      |
| -------------------------------- | :---: | :--------------: | :--------: |
| Layer 1 (bare prompt, no schema) | 56.5% | [49.57, 63.18] | 113 / 200  |
| Layer 1 + 2 (strict prompt)      |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| Layer 1 + 2 + 3 (evidence-bound) |  0.0% | [ 0.00,  1.88] |   0 / 200  |
```

## Reading

- The bare narrator fabricates at 56.5% on this task/model — far above the
  informal single-digit rates seen in earlier capstone trials on a different
  model. Fabrication propensity is strongly model-dependent.
- The strict prompt alone drove the observed rate to 0/200 on this model;
  Layer 2 and Layer 3 are empirically indistinguishable at N = 200.
- Layer 3's contribution is therefore the categorical worst-case guarantee
  (every returned citation ∈ evidence, by construction), not a lower observed
  rate. The 0/200 still leaves a Wilson upper bound of 1.88%; the contract is
  what closes that tail structurally.
- Composition of Layer-1 violations (invented names vs. real-but-out-of-evidence
  columns such as y_true/y_pred) was not logged in this run.
