# A/B scoring — ab-20260917-f98e3c-44031-control+revise+rubric-deepseek-flash-r3

Dataset: openml-44031 (california), arm: control+revise+rubric
Status: llm_failed, script exit code: 
Temperature: requested 1.0 (ab_protocol.md §5), status failed, in force unknown

## Rubric revision round (ab_protocol.md §9 A17)

- Turns: 1 (first turn identical to the `control` arm's)
- Second turn carries no mlcompass content: the six scored concerns are named as a generic rubric, with no claim about which of them this script has
- Pre-revision defect count: n/a of 6
- Post-revision defect count: n/a of 6 (this is the scored script)

## holdout_score (ab_protocol.md section 3)

Split: 15480 train / 5160 holdout rows at fraction 0.25, seed 2, stratified False, grouped on duplicate rows True.
Holdout rows present verbatim in train: 0 (§9 A4 holds this at 0).

Blank. model call failed: HTTP 402: Insufficient Balance

## process_defects (ab_protocol.md section 4)

Not scored: there is no emitted source to score. Section 4's checklist is a
property of the emitted script, and a run that produced none has no script.

Scored by a frozen function over the emitted source. No model scored this.
