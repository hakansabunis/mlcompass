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
