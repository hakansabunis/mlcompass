## Three-channel violation rates (live, task=synthetic, N = 200)

### Entity-fab rate
| Layer                   | Rate  | Wilson 95% CI    | k / N      |
| ----------------------- | :---: | :--------------: | :--------: |
| L1 bare prompt          |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| A-CONTRACT bare+TierA+TierB |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| STRESS bare+TierB only  |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| BASE Guardrails (stock) |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| BASE Guardrails (+TierB) |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| BASE strict, no enum    |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| BASE strict, stale enum |  0.0% | [ 0.00,  1.88] |   0 / 200  |

### Value-fab rate
| Layer                   | Rate  | Wilson 95% CI    | k / N      |
| ----------------------- | :---: | :--------------: | :--------: |
| L1 bare prompt          |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| A-CONTRACT bare+TierA+TierB |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| STRESS bare+TierB only  |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| BASE Guardrails (stock) |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| BASE Guardrails (+TierB) |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| BASE strict, no enum    |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| BASE strict, stale enum |  0.0% | [ 0.00,  1.88] |   0 / 200  |

### Omission rate
| Layer                   | Rate  | Wilson 95% CI    | k / N      |
| ----------------------- | :---: | :--------------: | :--------: |
| L1 bare prompt          |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| A-CONTRACT bare+TierA+TierB |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| STRESS bare+TierB only  |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| BASE Guardrails (stock) |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| BASE Guardrails (+TierB) |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| BASE strict, no enum    |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| BASE strict, stale enum |  0.0% | [ 0.00,  1.88] |   0 / 200  |

### Tier B catches (deterministic rejections that fired)
| Layer                   | Responses with >=1 catch | Total catches |
| ----------------------- | :----------------------: | :-----------: |
| L1 bare prompt          |         0 / 200    |         0     |
| A-CONTRACT bare+TierA+TierB |         0 / 200    |         0     |
| STRESS bare+TierB only  |         0 / 200    |         0     |
| BASE Guardrails (stock) |         0 / 200    |         0     |
| BASE Guardrails (+TierB) |         1 / 200    |         1     |
| BASE strict, no enum    |         0 / 200    |         0     |
| BASE strict, stale enum |         0 / 200    |         0     |

Layer 3 routes through the shipped investigate_leakage_bound: column enums are bound to the evidence at call time and Tier B deterministically verifies entity soundness, claim values (tolerance 0.005), and completeness; persistent omissions are flagged. The STRESS arm runs the same shipped contract with the BARE prompt and WITHOUT Tier A enums — its user-facing rates plus its catch counts show Tier B doing the work alone. See paper Section III.
