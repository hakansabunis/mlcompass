# RQ6 working notes — the second evidence-closed task

Data so far (deepseek-chat, evidence `bcaea394`, 24 columns / 13 statistics /
196 measured quantities, anchor `cat_feature_07`).

## Floor arm, P-L1, N=200, complete

    entity      0/200    0.0 %  [0.0, 1.9]
    value      17/200    8.5 %  [5.4, 13.2]
    omission    0/200    0.0 %  [0.0, 1.9]
    Tier B catches: 0     (correct: no Tier B on this arm)

3,277 structured claims scored, 25 offending.

## The decomposition, and it reproduces RQ1 on a different channel

**12 misquotes. 11 of them are misfiled, 1 invented.**

       8x  cat_feature_07 -> num_feature_07   (missing_pct)
       2x  cat_feature_07 -> num_feature_07   (missing_count)
       1x  cat_feature_06 -> num_feature_06   (missing_pct)
       1x  a value appearing nowhere in E

Every misfiled value is a *real measurement of another column*, and every one
confuses two columns that share a numeric suffix and differ in prefix. The
dominant source is `cat_feature_07` — which is the completeness anchor, the
column with the most missing data, therefore the most discussed.

This is the RQ1 finding on a task RQ1 never touched, and on a different
channel:

| | leakage | profile |
|---|---|---|
| channel | entity | value |
| misfiled | 162 / 162 | 11 / 12 |
| invented | 0 | 1 |

Leakage misfiles a real *name* into the wrong field. Profile misfiles a real
*measurement* onto the wrong entity. Same failure, relocated by the shape of
the evidence.

**13 unmeasured statistics.**

      12x  class_balance / class_balance_0 / class_balance_1
       1x  median

The twelve are genuine invention of a quantity kind: the profiler computes no
class balance, and the narrator wants to report one. The one `median` is
different and worth its own sentence — the profile carries `q50`, which *is*
the median, so this is a correct synonym the domain does not admit. That is
the aliasing question a reviewer asked, answered with data: on 3,277 claims it
happens once.

## What this means for the framing

The rate is 8.5 %, not leakage's 42 %. The second task fails, and fails much
more weakly. That is the honest headline and it should be stated before the
decomposition, not after it.

What survives at full strength is the *kind* of failure, which is the claim
RQ1 actually makes. A single blended "unfaithfulness rate" on this task would
read 8.5 % and hide that 11 of its 12 numeric errors are real measurements on
the wrong column — a failure an engineer reading the narration cannot detect,
because every number in it is a number the profiler really produced.

## Still to come

`tier_a` running (18/200, 1 value flag so far — Tier A cannot constrain a
float, so this is expected and is the point). Then `contract`, `stress`.
