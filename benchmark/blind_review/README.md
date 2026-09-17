# Blind rating of the six-item checklist

This is the check the paper cannot do for itself. Section 10 reports six
defects in instruments we wrote, and the differential audit in the same section
can only establish that two implementations *by the same authors* agree. What
it cannot establish is whether the checklist measures what it claims. That
needs raters who are not us.

**Nothing here has been run.** The apparatus is built and the sample is frozen;
the ratings are missing, and the paper says so rather than reporting a number.

## If you are a rater

You need `sample/` and nothing else. Do not open
`UNBLINDING_KEY_do_not_open_before_rating.csv` — it is one directory up for
exactly that reason.

1. Copy `sample/rating_sheet.csv` to `rater_<yourname>.csv`.
2. For each `S0NN.py` in `sample/`, read the script and fill one row. Write
   `1` if the defect is present, `0` if it is not, and leave the cell blank if
   you cannot tell. A blank is a real answer and is handled; a guess is not.
3. Do not discuss a script with the other rater until both sheets are done.

Each row gives you the three facts you need and cannot get from the script:
the target column's name, how many duplicate rows the input file contained,
and the minority-class fraction (blank for regression).

### The six rules

| Rule | Present when |
|---|---|
| `no_seed` | No seed reaches any framework the script imports, so re-running it gives different results. |
| `leak_fit_before_split` | A transformer or model is fitted on data that includes the test rows — before the split, or on the whole frame. |
| `leak_duplicate_rows` | The input contains duplicate rows and the script does not remove them before the data is split. If `input_has_duplicate_rows` is 0, this is always `0`. |
| `wrong_metric_for_imbalance` | Accuracy is reported on a target whose minority class is under 20 %. If `minority_fraction` is blank or ≥ 0.20, this is always `0`. |
| `no_validation` | The script never evaluates on data it did not train on — no held-out split, no cross-validation, no score on unseen rows. |
| `target_in_features` | The target column is still inside the feature matrix that is trained on. |

Judge the code as written. A comment claiming something the code does not do
is not a defence, and a defect the script works around by accident is still
absent.

## What was removed from these scripts, and why

Arm, model and dataset are stripped, the files are renamed `S001.py`
upward, and the order is shuffled under a fixed seed.

That was not enough. Two scripts named their own treatment in a comment —
`# Remove duplicate rows as advised` and
`# Hold-out validation split (for audit compliance ...)` — which tells a rater
the arm as plainly as a filename would. Both scripts turned out to be from
treatment arms, so the leak had a direction. Any occurrence of *advise*,
*audit*, *mlcompass*, a run directory or a temp path is now replaced by a
placeholder, and the generator re-reads its own output and refuses to write a
sample with a surviving tell. Every redaction is logged per script in the
unblinding key.

Redaction touches comments and string literals only. No statement a rule is
scored on was altered.

## After both sheets are in

```bash
python benchmark/blind_review/score_blind_review.py rater_a.csv rater_b.csv
```

Reports, per rule: Cohen's κ between the raters, and the checklist's precision
and recall against the human reference. Where the raters disagree with each
other, that rule is excluded from precision and recall for that script — a
reference the raters cannot agree on is not a reference, and averaging them
into one would hide the ambiguity worth reporting.

## Regenerating the sample

```bash
python benchmark/blind_review/make_blind_sample.py --per-cell 2
```

24 scripts, 6 per arm, stratified across (arm, dataset), seed 20260917.
Regenerating invalidates any ratings already collected against the old sample,
because the blind ids are positional.
