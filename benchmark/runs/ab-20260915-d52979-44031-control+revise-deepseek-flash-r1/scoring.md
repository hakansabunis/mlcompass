# A/B scoring — ab-20260915-d52979-44031-control+revise-deepseek-flash-r1

Dataset: openml-44031 (california), arm: control+revise
Status: no_code_revision, script exit code: 
Temperature: requested 1.0 (ab_protocol.md §5), status sent, in force 1.0

## Self-revision round (ab_protocol.md §9 A8)

- Turns: 2 (first turn identical to the `control` arm's)
- Second turn carries no mlcompass content: it asks the model to review and fix its own script, naming no concern
- Pre-revision defect count: 1 of 6
- Post-revision defect count: n/a of 6 (this is the scored script)

## holdout_score (ab_protocol.md section 3)

Split: 15480 train / 5160 holdout rows at fraction 0.25, seed 20260915, stratified False, grouped on duplicate rows True.
Holdout rows present verbatim in train: 0 (§9 A4 holds this at 0).

Blank. no revised script to run: no Python found in the reply

## process_defects (ab_protocol.md section 4)

Not scored: there is no emitted source to score. Section 4's checklist is a
property of the emitted script, and a run that produced none has no script.

Scored by a frozen function over the emitted source. No model scored this.
