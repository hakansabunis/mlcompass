# Where this stopped, 2026-09-19

Paused mid-battery to shut the machine down. Nothing is in a broken state; the
only loss is the API calls the killed run had already paid for and not yet
written, which is a harness flaw worth fixing before the next run (see below).

## Run this first

```bash
source scripts/load_keys.sh
python -X utf8 -u scripts/reproduce_profile_battery.py \
  --arm bare --arm tier_a --arm contract --arm stress \
  --n 200 --provider deepseek
```

About 70 minutes, 800 calls on `deepseek-chat`. It refuses to overwrite any arm
whose file already exists, so it is safe to re-run after an interruption — but
only whole arms are saved, so an interrupted arm restarts from zero.

**Fix before running if there is time:** the battery writes each arm's `.jsonl`
once, after all 200 calls. An interrupted arm therefore loses every response it
paid for. Writing incrementally (append per response, or every 20) would have
saved two partial arms today.

Do **not** switch models. The leakage battery ran on `deepseek-chat`, and the
whole point of the second task is that a difference between the two tasks is a
difference in the tasks. `deepseek-v4-pro` was offered and declined for this
reason.

## What the battery is for

Queue item 7, the second evidence-closed task: two reviewers independently
objected that the paper defines a class and measures one member. This narrates
`analyze_dataset` through the same `evidence_contract` verifier, no branch
naming either task. 24 columns / 13 statistics / 196 measured quantities
against leakage's 10 / 1 / 10.

Four arms, named to match the leakage battery so a difference is a difference
in the tasks: `bare` (P-L1), `tier_a` (P-TIER-A-ONLY), `contract` (P-CONTRACT),
`stress` (P-STRESS).

## What is already known, and it is the good news

The floor arm fails, and **on a different channel than leakage**. The last
measured run (now superseded for a separate reason) read:

    entity      1/200    0.5 %
    value      45/200   22.5 %
    omission    0/200    0.0 %

Leakage fails on entity at 42 % (misfiling `r2` into a column field). Profile
fails on value. So the class is real and the dominant failure *within* it
depends on the evidence shape — which supports the paper's thesis harder than
a second 42 % would have.

Expect the corrected run to read lower than 22.5 %: replaying the preserved
responses under the fixed domain moved the value channel from 71 offending
claims to 46. Whatever it reads is the number.

## Two instrument faults found today, both ours, both fixed

**Ninth — the floor arm was repaired before it was scored.** `strip_unsound`
ran unconditionally, so 60 of 200 bare responses had their violations deleted
and the arm read 0/200 on a task that fails at 30 %. The tell was "Tier B
catches: 60" printed for an arm with no Tier B. Fixed by `verify_response`,
which now gates the retry loop and the strip together. `raw_columns` and
`raw_claims` are preserved so this can never again be invisible.

**Tenth — the domain renamed what the evidence called things.** The profiler
writes `outliers: {iqr_count, z_score_count}`; the binder called them
`iqr_outliers` and `z_score_outliers`. Narrators wrote the evidence's own word
and scored as inventing. About a third of the measured failure was our
vocabulary.

The real fault there is a design one and belongs in the paper: **the claim
schema is flat and the evidence was nested.** Handed a nested quantity and a
flat `(column, statistic, value)` slot, narrators invented five different
flattenings — `z_score_count`, `outliers.z_score_count`, `outliers`,
`iqr_count`, `outliers_iqr_count`. No domain can anticipate that. `compact()`
now lifts them, so the name the narrator reads is the name the domain admits.
The evidence hash changed to `bcaea394` as a result.

## Then, in order

1. **Write RQ6.** The second task, as a compact section — not a parallel
   repeat of RQ1–RQ4. EMSE is already 51 pp. and TSE 17; the advisor called 38
   too long. TSE gets it tighter still.
2. **Fold in the scalability numbers**, already measured and free:

   | columns | prompt | schema |
   |---|---|---|
   | 10 | 3,932 B | 1,515 B |
   | 100 | 33,763 B | 4,738 B |
   | 500 | 178,434 B | 19,920 B |
   | 1000 | 358,107 B | 38,920 B |

   §12 worries a thousand-element enum will hit provider limits. It is aimed
   at the wrong thing: the enum costs ~38 bytes per column, the evidence that
   produced it ~358. Tier A's cost is a tenth of what you must carry anyway.
   Reproduce with `--dry-run --frame-columns N --max-columns N`, no API spend.
3. **Fold in the mutation tests** (`tests/test_tier_b_mutations.py`, 36 tests).
   They answer "Tier A caught everything, what is Tier B for" with an argument
   rather than an anecdote: value is a predicate over a continuum and
   completeness is a property of the whole response, so two of three channels
   cannot be expressed in any schema.
4. **Say plainly what the paper establishes.** It is accumulating negative
   results — RQ5's attribution withdrawn, RQ2/3 narrowed, ten instrument
   faults, free-text unmeasurable — and the self-corrections are drowning the
   claim that survives: *for evidence-closed narration, binding the admissible
   set to the evidence at call time prevents fabrication on the validated
   channels by construction, behind a black-box API.* That needs to be stated
   loudly and early.
5. Recompile both, verify tables and figures again, push.

## Settled, do not reopen

- **#1b blind audit: not in the paper.** Sheets that arrived under human names
  were Gemini agents; they are filed honestly under
  `benchmark/blind_review/pilot_llm/` with the reasoning. §12 continues to say
  the audit is built and unrun, which is true. The apparatus is fixed and
  usable by real raters: the key now scores the redacted script with the
  current checker, and blind ids are unchanged.
- **Repo is private with no Zenodo DOI.** The paper says open-source and gives
  the URL. Not a research task, but a submission blocker.

## Background note

`nohup` does not survive in this environment. Two runs died silently with empty
logs before this was noticed. Use the harness's own background tracking.
