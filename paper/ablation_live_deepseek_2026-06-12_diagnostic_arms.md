# Diagnostic arms record — Tier A isolation + full contract under worst paraphrase

- **Date:** 2026-06-12 (same session as the battery record)
- **Command:** `python scripts/reproduce_hallucination_ablation.py --mode live --provider deepseek --n 200 --only-extra`
- **Provider/model:** deepseek-chat, provider-default sampling; synthetic task, seed 0
- **Arms:**
  - `tier_a`: evidence-bound enum IN the tool schema, bare prompt, **no Tier B verification** (single call, raw scoring)
  - `stress_mech`: the FULL shipped contract (Tier A enums + Tier B) under the sweep's worst
    naturally occurring paraphrase (`mechanical`, 100/100 fabrication with the open schema)

## Results (verbatim)

| Arm | Entity-fab | Value-fab | Omission | Tier B catches |
|---|---|---|---|---|
| TIER-A only (enum, no verify) | 0.0% (0/200) [0.00, 1.88] | 0/200 | 0/200 | n/a (verification off) |
| L3 + worst paraphrase | 0.0% (0/200) [0.00, 1.88] | 0/200 | 0/200 | 0 (no first-attempt violations) |

## Reading

- **Tier A isolated:** from L1's 11.5% baseline (open schema, same bare prompt) to 0/200 with
  the enum merely present in the schema. On this endpoint the enum acts as strong
  *steering* (schema-as-instruction), even without documented decode-time enforcement.
- **Full contract under the worst paraphrase:** the `mechanical` wording fabricates 100/100
  with an open schema (sweep), yet produced zero first-attempt violations under the
  contract: the enum steering absorbed the worst wording outright, so Tier B had nothing
  to catch.
- **Honest scope:** both arms ran on the synthetic task only; the harder real-data task
  (43.5% baseline) was not repeated for these arms. Steering, like prompting, is an
  empirical property and can regress; it is not enforcement. The certifiable layer
  remains Tier B.
- Combined defense-in-depth picture across all runs: prompts can reach 0 (but span
  1%–100% across paraphrases), the enum reaches 0 (but is uncertifiable steering),
  Tier B demonstrably catches and corrects when both fail (76 catches, 0 through, on
  real data).
