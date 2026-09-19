# Superseded: the floor arm was repaired before it was scored

`deepseek_deepseek-chat_profile_bare_n200_seed0_e8d80f33c.jsonl`, run
2026-09-18, is preserved here rather than deleted because it is evidence about
the instrument.

The `bare` arm carries no Tier A enum and no Tier B verification. It is the
floor: what reaches a user when nothing enforces anything. The first version of
`narrate_profile_bound` called `strip_unsound` unconditionally, so every
unsound citation and every misquoted claim was deleted from the payload before
the scorer saw it.

The arm reported:

    entity     0/200   0.0 %  [0.0, 1.9]
    value      0/200   0.0 %  [0.0, 1.9]
    omission   1/200   0.5 %  [0.1, 2.8]
    Tier B catches: 60

The last line is the tell. A Tier B catch on an arm that has no Tier B is a
contradiction, and the 60 is the count of responses whose violations were
silently repaired: 57 value, 2 entity, 1 value+omission. The floor arm does not
fail at 0 %; it fails on at least 60 of 200 responses, and the harness produced
exactly the number the hypothesis would have liked.

What cannot be recovered from this file is the decomposition. The record keeps
the stripped `columns` and `claims`, so the names that were deleted -- and
therefore whether each was invented or misfiled -- are gone. Only the rejection
counters survive. That is why `raw_columns` and `raw_claims` are now recorded
alongside the clean ones, and why this run is re-done rather than rescored.

Fixed by `verify_response`, which gates the retry loop *and* the strip. The
violations are still computed on every arm, because the diagnostic arms need to
know what Tier B would have caught; what changes is whether anything is done
about them.

This is the ninth fault found in our own instruments, and the second whose
direction favoured the hypothesis we are arguing for.

---

# Superseded: the statistic domain renamed what the evidence called things

`deepseek_deepseek-chat_profile_bare_n200_seed0_e8d80f33c.jsonl` (the corrected
re-run of the above) and `prefix_tier_a_n200.jsonl` are preserved here for the
same reason: they measure a domain that was not bound to the evidence.

The profiler writes `outliers: {iqr_count: ..., z_score_count: ...}`. The
binder called those `iqr_outliers` and `z_score_outliers` because they read
better. So a narrator writing `z_score_count` -- the evidence's own word --
scored as naming a quantity the evidence does not carry.

The run reported:

    entity      1/200    0.5 %
    value      45/200   22.5 %  [17.3, 28.8]
    omission    0/200    0.0 %

Replaying the preserved responses under the corrected names moves the value
channel from 71 offending claims across 38 responses to 46 across 29. Roughly a
third of the measured failure was the instrument's vocabulary, not the
narrator's.

The deeper fault is a design one, and it is why the fix is not just a rename.
The claim schema is flat -- `(column, statistic, value)` -- and the evidence was
nested. A narrator handed a nested quantity and a flat claim slot has to invent
a flattening, and narrators invented five: `z_score_count`,
`outliers.z_score_count`, `outliers`, `iqr_count`, `outliers_iqr_count`. No
domain can anticipate that set. `compact()` now lifts the outlier counts to the
top level so the name the narrator reads is the name the domain admits, which
is what binding a domain to evidence is supposed to mean.

That changes the evidence hash, so the corrected battery runs under a new
`e` prefix and cannot collide with these.

Tenth fault found in our own instruments.

---

# Discarded before it was finished: the contract arm had a prompt confound

`partial_contract_strictprompt_n101.jsonl` is 101 responses of a `contract` arm
that was given the strict faithfulness prompt while `bare`, `tier_a` and
`stress` were given the bare one.

That is the confound the leakage battery's `A-CONTRACT` exists to avoid. Its
definition in the design section is "the shipped enforcement stack with the
faithfulness *prompt removed*, so the contract is compared to bare-prompt
baselines without confounding prompt with mechanism". The profile battery
reintroduced it in one line.

Caught at 101 of 200, before any number from it was reported, so this is a
design error in a new experiment rather than a fault in a published
instrument — it is not counted among the ten. The run is kept because it is a
valid measurement of something else: strict prompt plus Tier A plus Tier B,
which is the configuration the tool actually ships. It is simply not the arm
the comparison needs, and mixing the two would make the contract arm look
better for a reason that is not the contract.

Every arm now gets `PROFILE_BARE_PROMPT`.
