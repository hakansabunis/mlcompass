# Not the blind audit. Two LLM agents, kept for the record.

These two sheets were produced by Gemini agents, not by human raters, and they
were originally filed under human names. They are moved here and renamed so
nobody — including us, later — mistakes them for the audit `../README.md`
describes.

**They do not answer the question the audit exists to answer.** That question
is: both scorers are ours, written in the same week by people who had read each
other's, so what if they share a blind spot? Two language models reading the
same six-rule README and the same scripts is not an escape from that structure.
It is the same structure with two more instances, drawn from overlapping
training data and handed identical framing. And the audit asks whether the
checklist matches *human* judgement of what a defect is; model judgement is a
different quantity.

Nothing from these sheets is reported in the manuscript, and §12 continues to
say the blind audit is built and unrun, because it is.

## What they contain, for anyone who wants to look

Scored against the CURRENT checklist (the frozen key was stale — see below):

| rule | κ between agents | agents' positives | checklist | TP | FN |
|---|---|---|---|---|---|
| `leak_duplicate_rows` | 1.000 | 11 | 11 | 11 | 0 |
| `leak_fit_before_split` | 0.778 | 2 | 0 | 0 | 2 |
| `no_validation` | 0.429 | 2 | 2 | 2 | 0 |
| `no_seed` | n/a | 0 | 0 | — | — |
| `wrong_metric_for_imbalance` | n/a | 0 | 0 | — | — |
| `target_in_features` | n/a | 0 | 0 | — | — |

Three of six rules carry no positives on either side, so they carry no
information. The remaining three rest on 11, 2 and 2 positives. A precision of
1.00 computed on two true positives is not a measurement, and the exclusion
rule makes it worse: `no_validation`'s four disagreements are dropped before
precision is computed, and those four are exactly where the ambiguity lives, so
what survives is the easy subset.

## The one lead, run down and closed

The agents flagged `leak_fit_before_split` on S005 and S019 where the checklist
does not, which would be a miss worth reporting if it were one. It is not.
Neither script splits the data at all — no `train_test_split`, no `KFold`, no
cross-validation — so there is no split for a fit to precede. The checklist
catches both scripts under `no_validation` and `leak_duplicate_rows`. The
disagreement is definitional, not a defect: when there is no split, the fault
is that nothing was held out, and that is the rule that fires.

## A real bug this did surface

The frozen key was generated before amendment A16 and carried
`target_in_features: true` on S007, S013, S017 and S021 — the four false
positives A16 removed. Anyone running `score_blind_review.py` as shipped would
have read "precision 0.00" and rediscovered a defect the paper already reports
and already fixed. That is a fault in the apparatus, not in the checklist, and
it is fixed separately.
