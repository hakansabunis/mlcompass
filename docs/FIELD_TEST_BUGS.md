# Field-Test Bugs — Per-Bug Detail

Reviewer #1 of the mlcompass paper pointed out that lumping eleven
field-test bugs into a single "11 bugs caught" headline understates
how trivial some of them were. Five of the eleven were missing target
names from a hard-coded list, which any practitioner could fix in
sixty seconds. Three were genuine CLI/MCP state-parity gaps. Three
were UX corner cases.

This document records each bug, its category, the commit that fixed
it, the CHANGELOG entry, and the regression test that locks it in. A
reader who wants to gauge engineering significance can decide for
themselves.

For aggregate counts and the paper's headline table, see Table II of
the paper. For threat-model gaps the field tests *did not* surface,
see `docs/THREAT_MODEL.md`.

---

## Bug categories

| Code  | Category                | Description                                                                 |
| ----- | ----------------------- | --------------------------------------------------------------------------- |
| TN    | Target Name             | A canonical Kaggle target column name was missing from a hard-coded list.   |
| SP    | State Parity (CLI/MCP)  | CLI and MCP server diverged in what they wrote to the project ledger.       |
| CR    | Crash                   | A code path threw an unhandled exception on a real dataset.                 |
| UX    | UX Confusion            | An automatic heuristic surfaced a misleading warning on a corner-case input.|

The categorization is ours. We do not believe TN bugs deserve the same
weight as SP or CR bugs, but we report them all because honesty about
the field-test cadence requires it.

---

## FT2-1 — Ames House Prices, missing `saleprice` target

- **Category.** TN
- **Patch release.** v0.7.0
- **Symptom.** `mlcompass advise train.csv` printed *"no target
  column detected — passing target via --target may be required"* on
  the Ames House Prices dataset. The column `SalePrice` was the
  obvious target.
- **Root cause.** The high-confidence regression target list in
  `tools/dataset.py` contained `price`, `target`, and `cost` but
  not `saleprice`, `sale_price`, or `house_price`.
- **Fix.** Extended the list to include `saleprice`, `sale_price`,
  `house_price`, `home_price`, `listing_price`, `asking_price`,
  `purchase_price`, `valuation`, `appraisal`, and `assessed_value`.
- **Test.** `tests/test_tools_dataset.py::test_kaggle_target_names`.
- **Significance.** Low. Anyone using the tool could have worked
  around it by passing `--target SalePrice`. The fix is a list
  extension.

## FT2-2 — Ames House Prices, year column tagged as ID

- **Category.** UX
- **Patch release.** v0.7.0
- **Symptom.** `mlcompass advise train.csv` warned that `YearBuilt`
  looked like an ID column because most values were unique within
  the visible sample.
- **Root cause.** The ID-column heuristic in `tools/dataset.py`
  fired on any column with >85% unique values, without checking
  whether the column name was a year-like canonical name.
- **Fix.** Added a year-name early-exit list (`year`, `yr`,
  `yearbuilt`, `yearsold`, etc.) that suppresses the ID warning.
- **Test.** `tests/test_tools_dataset.py::test_year_column_not_flagged_as_id`.
- **Significance.** Low. Warning rather than a crash.

## FT2-3 — Ames House Prices, sparse-column IQR crash

- **Category.** CR
- **Patch release.** v0.7.0
- **Symptom.** `mlcompass advise train.csv` threw a `ValueError:
  cannot convert float NaN to integer` when scanning the `PoolQC`
  column, which was 96% NaN.
- **Root cause.** The IQR computation in the warning formatter
  attempted to convert the IQR result to an integer for display,
  but a column with most values missing produced `nan`.
- **Fix.** Wrapped the IQR computation in a sparse-skip guard
  (`if col.isna().mean() > 0.9: skip`).
- **Test.** `tests/test_tools_dataset.py::test_sparse_column_skipped`.
- **Significance.** Medium. A crash that would have stopped the
  pipeline on any dataset with a sparse column.

## FT3-1 — Titanic, missing `survived` target

- **Category.** TN
- **Patch release.** v0.7.1
- **Symptom.** `Survived` was identified as the target but at
  "low confidence" only.
- **Root cause.** The binary-target high-confidence list contained
  `churn`, `fraud`, `target`, and a handful of others, but not
  `survived` or related medical/financial outcome names.
- **Fix.** Extended the list to include `survived`, `outcome`,
  `diagnosis`, `defaulted`, `approved`, `bankrupt`, `default`,
  `subscribed`, `renewed`, `passed`.
- **Test.** `tests/test_tools_dataset.py::test_binary_target_names`.
- **Significance.** Low.

## FT4-1 — Penguins, multiclass mis-classified as binary

- **Category.** CR (logic bug, would crash downstream evaluation)
- **Patch release.** v0.7.2
- **Symptom.** `mlcompass advise penguins.csv --target species`
  reported *"binary classification"* on the three-class species
  column.
- **Root cause.** The class-count heuristic in
  `tools/dataset.py::_infer_task` read `nunique() == 2` instead of
  comparing the class count to a binary/multiclass threshold.
- **Fix.** Rewrote the heuristic to detect binary, multiclass, and
  regression correctly; added a `task_type` enum.
- **Test.** `tests/test_tools_dataset.py::test_multiclass_detected`.
- **Significance.** **High.** Would have caused downstream
  `evaluate` to compute binary metrics on a multiclass output —
  the kind of silent wrong-answer that the tool exists to prevent.

## FT4-2 — Penguins, MCP server bypasses project ledger

- **Category.** SP
- **Patch release.** v0.7.2
- **Symptom.** A full pipeline run through the MCP server (Claude
  Code as client) left `mlcompass_status` showing an empty
  decisions list.
- **Root cause.** The CLI's Click handler wrapped each subcommand
  in a "write to ledger" step. The MCP server's tool functions
  called the deterministic tools directly and silently bypassed
  the ledger write.
- **Fix.** Introduced `_persist_to_ledger` in
  `src/mlcompass/mcp_server.py` and made every MCP tool function
  call it. Added CLI/MCP parity tests.
- **Test.** `tests/test_mcp_server.py::test_advise_writes_to_ledger`,
  `tests/test_mcp_server.py::test_cli_mcp_parity_after_full_pipeline`.
- **Significance.** **High.** This was the most generalizable
  finding of the field tests. Documented in paper Section III.D
  as a first-class methodological lesson.

## FT4-3 — Penguins, multiclass target name list incomplete

- **Category.** TN
- **Patch release.** v0.7.2
- **Symptom.** `species` was not on the multiclass high-confidence
  target list.
- **Root cause.** The multiclass list was, at the time of v0.7.1,
  empty. We had not anticipated multiclass corpus-style target
  names at all.
- **Fix.** Added a multiclass list: `species`, `category`, `class`,
  `label`, `type`, `genre`, `cluster`.
- **Test.** `tests/test_tools_dataset.py::test_multiclass_target_names`.
- **Significance.** Low.

## FT5-1 — Insurance Charges, pre-init activity hint missing

- **Category.** UX
- **Patch release.** v0.7.3
- **Symptom.** A user who calls MCP tools *before* running
  `mlcompass init` will see those tool calls silently dropped (by
  design — we don't auto-init). After the user finally runs
  `init`, the project ledger is empty and there is no indication
  that the prior MCP calls were lost.
- **Root cause.** No proactive hint surfaced this state.
- **Fix.** Added a hint in `tools/runs.py::status` that warns when
  the ledger contains only the `init` decision and no others.
- **Test.** `tests/test_tools_runs.py::test_pre_init_hint`.
- **Significance.** Low. A discoverability fix.

## FT5-2 — Insurance Charges, missing `charges` target

- **Category.** TN
- **Patch release.** v0.7.3
- **Symptom.** `charges` was detected as a regression target but
  at "low confidence" because the high-confidence list was missing
  finance and insurance target names.
- **Root cause.** The regression target list, even after v0.7.0's
  extension, was biased toward housing/property names. It missed
  the entire finance / insurance / billing space.
- **Fix.** Added `charges`, `total_charges`, `medical_cost`, `fee`,
  `tuition`, `expenses`, `arpu`, `spend`, `revenue`, `amount`.
- **Test.** `tests/test_tools_dataset.py::test_finance_target_names`.
- **Significance.** Low.

## FT5-3 — Insurance Charges, MCP active-state fields null

- **Category.** SP
- **Patch release.** v0.7.3
- **Symptom.** After a full pipeline run through MCP,
  `mlcompass_status` showed `project_type`, `target_column`, and
  `active_dataset` as `null` even though MCP `advise` had
  succeeded.
- **Root cause.** The v0.7.2 fix (FT4-2) made the MCP wrappers
  call `_persist_to_ledger`, but the helper only appended to
  `decisions[]` and `advice.log`. It did not update the
  active-state fields, which the CLI's Click handler had been
  updating separately.
- **Fix.** Extended `_persist_to_ledger` with a `state_updates`
  parameter that calls `ProjectContext.write_context` to update
  the active state. Added a regression test that catches this
  specific kind of parity drift.
- **Test.** `tests/test_mcp_server.py::test_advise_updates_active_state`.
- **Significance.** **High.** Same family as FT4-2. Confirmed that
  CLI/MCP parity needs to be tested as a first-class invariant
  per command, not handled with a single shared helper.

---

## Aggregate Summary

| Category    | Count | Severity weighting          |
| ----------- | :---: | --------------------------- |
| Target Name | 5     | Low (list extensions)        |
| State Parity| 3     | High (generalizable lesson)  |
| Crash       | 2     | Medium-High (would stop user)|
| UX          | 3     | Low (warnings, no crashes)   |

When the paper reports "eleven bugs caught", these are the eleven.
The three SP bugs and the FT4-1 multiclass crash are the four that
materially affected the trustworthiness of the tool. The remaining
seven are list extensions and UX polish that any open-source tool
accumulates with use.
