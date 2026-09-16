## Three-channel violation rates (live, task=synthetic, N = 200)

### Entity-fab rate
| Layer                   | Rate  | Wilson 95% CI    | k / N      |
| ----------------------- | :---: | :--------------: | :--------: |
| TIER-A only (enum, no verify) |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| L3 + worst paraphrase   |  0.0% | [ 0.00,  1.88] |   0 / 200  |

### Value-fab rate
| Layer                   | Rate  | Wilson 95% CI    | k / N      |
| ----------------------- | :---: | :--------------: | :--------: |
| TIER-A only (enum, no verify) |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| L3 + worst paraphrase   |  0.0% | [ 0.00,  1.88] |   0 / 200  |

### Omission rate
| Layer                   | Rate  | Wilson 95% CI    | k / N      |
| ----------------------- | :---: | :--------------: | :--------: |
| TIER-A only (enum, no verify) |  0.0% | [ 0.00,  1.88] |   0 / 200  |
| L3 + worst paraphrase   |  0.0% | [ 0.00,  1.88] |   0 / 200  |

### Tier B catches (deterministic rejections that fired)
| Layer                   | Responses with >=1 catch | Total catches |
| ----------------------- | :----------------------: | :-----------: |
| TIER-A only (enum, no verify) |         0 / 200    |         0     |
| L3 + worst paraphrase   |         0 / 200    |         0     |

Layer 3 routes through the shipped investigate_leakage_bound: column enums are bound to the evidence at call time and Tier B deterministically verifies entity soundness, claim values (tolerance 0.005), and completeness; persistent omissions are flagged. The STRESS arm runs the same shipped contract with the BARE prompt and WITHOUT Tier A enums — its user-facing rates plus its catch counts show Tier B doing the work alone. See paper Section III.
