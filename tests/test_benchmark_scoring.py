"""Defects found by a collaborator reviewing the first benchmark results.

Four defects, one section each. Every one of them is a case where the
recorded number and the thing it is supposed to measure came apart:

1. An `llm:*` run whose advisor produced no parseable reply was recorded as
   `completed` with a clean score, because the CLI still exited 0.
2. Recall was scanned over the whole terminal output, so every `llm:*` lane
   inherited the deterministic layer's detections and scored them as its own.
3. `unverified_findings` pooled suggestions (which have no ground truth and
   never will) with unmatched detection claims (which are the number worth
   having).
4. The pinned input hashes were taken over CRLF bytes, so the same logical
   file fails verification on a checkout with LF line endings.

The fixtures below are cut down from real preserved evidence under
`benchmark/runs/`; the two tests that read that evidence directly skip when
it is absent, so the suite stays runnable from a bare checkout.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmark"


def _load_runner():
    """Import `benchmark/run_benchmark.py` — it is a script, not a package."""
    spec = importlib.util.spec_from_file_location("bench_run_benchmark", BENCH / "run_benchmark.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rb = _load_runner()


# --------------------------------------------------------------------------- #
# Fixtures cut down from real runs                                            #
# --------------------------------------------------------------------------- #

CASES_1464 = [
    {
        "case_id": "1464-duplicates",
        "issue_id": "DUP-1464",
        "match_pattern": "duplicate row",
        "required_values": ["215"],
    }
]

# The deterministic layer prints before the LLM is ever called, so this text
# is present in every configuration including the ones under test.
DETERMINISTIC_PREFIX = """\
(no .mlcompass/ project found in current path - running standalone; results
will not be persisted)

⚠ Warnings
  • 215 exact duplicate row(s) (28.7% of the data). A random split puts copies
of the same row in both train and test, which inflates held-out scores.
"""

# An LLM region that never mentions the duplicates: the model was handed the
# finding and narrated other things instead.
LLM_REGION_SILENT = """
✨ Recommended models
┌───────────┬──────────┐
│ Model     │ Why      │
│ LogReg    │ baseline │
└───────────┴──────────┘

\U0001f527 Feature engineering
  • V3 → log1p(V3) or a rank/quantile transform
     V3 is heavily right-skewed with 45 IQR outliers.
  • V2, V4 → V2 / max(V4, 1) — donations per month of tenure
     A normalized intensity measure comparable across donors.

⚠ Pitfalls
  • Moderate class imbalance (Class 1 = 76.2%, Class 2 = 23.8%)
     → Use stratified CV and report AUC rather than raw accuracy.
  • Small sample (748 rows, 4 predictors); a single random split is noisy
     → Use repeated stratified k-fold and report mean +/- std.
"""

# The same run, but the model does repeat the duplicate finding in Pitfalls.
LLM_REGION_ECHOES = (
    LLM_REGION_SILENT
    + """\
  • 215 exact duplicate rows (28.7% of data); a random split leaks rows
     → De-duplicate before splitting, or use GroupKFold.
"""
)

# Verbatim tail of the advisor failure, from
# runs/exp-20260915-eb4c19-1480-llm-mistral-ministral-8b-r3/stdout.txt
ADVISOR_FAILED_TAIL = """
✗ Advisor returned an invalid response: Agent response was not valid JSON: '{
"models": [     {       "name": "Logistic Regression",       "reason":
"Baseline for interpretability and linear relationships; useful to establish a
sanity check before complex models (e.g., V1, '
"""

FAILED_RUN = BENCH / "runs" / "exp-20260915-eb4c19-1480-llm-mistral-ministral-8b-r3"


# --------------------------------------------------------------------------- #
# Defect 1 — a failed LLM run was scored as a clean success                    #
# --------------------------------------------------------------------------- #


class TestDefect1FailedAdvisorRun:
    def test_advisor_failure_line_is_detected(self):
        assert rb.advisor_failed(DETERMINISTIC_PREFIX + ADVISOR_FAILED_TAIL) is True
        assert rb.advisor_failed(DETERMINISTIC_PREFIX + LLM_REGION_SILENT) is False

    def test_llm_run_without_a_parseable_reply_is_not_completed(self):
        """Exit code 0 is not evidence the configuration under test ran."""
        status, note = rb.classify_status(
            "llm:mistral-ministral-8b",
            exit_code=0,
            stdout=DETERMINISTIC_PREFIX + ADVISOR_FAILED_TAIL,
        )
        assert status == "failed"
        assert "advisor" in note.lower()

    def test_a_healthy_llm_run_still_completes(self):
        status, note = rb.classify_status(
            "llm:mistral-ministral-8b",
            exit_code=0,
            stdout=DETERMINISTIC_PREFIX + LLM_REGION_SILENT,
        )
        assert status == "completed"
        assert note == ""

    def test_the_deterministic_lane_is_never_judged_on_the_advisor(self):
        """`deterministic` never calls the advisor; its status is the exit code."""
        status, _ = rb.classify_status(
            "deterministic", exit_code=0, stdout=DETERMINISTIC_PREFIX + ADVISOR_FAILED_TAIL
        )
        assert status == "completed"

    def test_a_nonzero_exit_still_fails(self):
        status, note = rb.classify_status(
            "llm:mistral-ministral-8b",
            exit_code=1,
            stdout=DETERMINISTIC_PREFIX + LLM_REGION_SILENT,
        )
        assert status == "failed"
        assert "exit" in note.lower()

    def test_the_real_preserved_run_is_classified_failed(self):
        if not FAILED_RUN.exists():
            pytest.skip("benchmark evidence not present in this checkout")
        stdout = (FAILED_RUN / "stdout.txt").read_text(encoding="utf-8")
        status, _ = rb.classify_status("llm:mistral-ministral-8b", exit_code=0, stdout=stdout)
        assert status == "failed"


# --------------------------------------------------------------------------- #
# Defect 2 — recall did not measure the LLM                                    #
# --------------------------------------------------------------------------- #


class TestDefect2ScopeOfScoring:
    def test_llm_lane_does_not_inherit_deterministic_detections(self):
        output = DETERMINISTIC_PREFIX + LLM_REGION_SILENT
        scored = rb.score(output, CASES_1464, config_id="llm:deepseek-flash")
        assert scored["correct_detections"] == 0
        assert scored["missed_issues"] == 1
        # Nothing is hidden: the inherited detection is still reported.
        assert scored["inherited_detected_ids"] == ["DUP-1464"]

    def test_llm_lane_is_credited_when_it_does_repeat_the_finding(self):
        output = DETERMINISTIC_PREFIX + LLM_REGION_ECHOES
        scored = rb.score(output, CASES_1464, config_id="llm:deepseek-flash")
        assert scored["correct_detections"] == 1
        assert scored["detected_ids"] == ["DUP-1464"]

    def test_deterministic_lane_still_scores_the_whole_output(self):
        scored = rb.score(DETERMINISTIC_PREFIX, CASES_1464, config_id="deterministic")
        assert scored["correct_detections"] == 1
        assert scored["scope"] == "full output"

    def test_scope_starts_at_the_recommended_models_panel(self):
        output = DETERMINISTIC_PREFIX + LLM_REGION_SILENT
        scoped = rb.scoring_scope(output, "llm:deepseek-flash")
        assert scoped.startswith("✨ Recommended models")
        assert "215 exact duplicate row" not in scoped

    def test_the_real_failed_run_scores_nothing_for_the_llm(self):
        """The 1/1 on this run was text the model never produced."""
        if not FAILED_RUN.exists():
            pytest.skip("benchmark evidence not present in this checkout")
        stdout = (FAILED_RUN / "stdout.txt").read_text(encoding="utf-8")
        cases = [
            {
                "case_id": "1480-duplicates",
                "issue_id": "DUP-1480",
                "match_pattern": "duplicate row",
                "required_values": ["13"],
            }
        ]
        assert rb.score(stdout, cases, config_id="deterministic")["correct_detections"] == 1
        assert (
            rb.score(stdout, cases, config_id="llm:mistral-ministral-8b")["correct_detections"] == 0
        )


# --------------------------------------------------------------------------- #
# Defect 3 — unverified_findings is not a hallucination measure                #
# --------------------------------------------------------------------------- #


class TestDefect3ClaimTypes:
    def test_claims_are_split_by_the_section_they_came_from(self):
        output = DETERMINISTIC_PREFIX + LLM_REGION_SILENT
        scored = rb.score(output, CASES_1464, config_id="llm:deepseek-flash")
        assert scored["unverified_suggestions"] == 2  # two feature-engineering bullets
        assert scored["unverified_detection_claims"] == 2  # two pitfalls

    def test_the_two_columns_sum_to_the_old_one(self):
        for region in (LLM_REGION_SILENT, LLM_REGION_ECHOES):
            scored = rb.score(
                DETERMINISTIC_PREFIX + region, CASES_1464, config_id="llm:deepseek-flash"
            )
            assert (
                scored["unverified_detection_claims"] + scored["unverified_suggestions"]
                == scored["unverified_findings"]
            )

    def test_a_matched_detection_claim_is_not_counted_as_unverified(self):
        silent = rb.score(
            DETERMINISTIC_PREFIX + LLM_REGION_SILENT,
            CASES_1464,
            config_id="llm:deepseek-flash",
        )
        echoes = rb.score(
            DETERMINISTIC_PREFIX + LLM_REGION_ECHOES,
            CASES_1464,
            config_id="llm:deepseek-flash",
        )
        # The extra pitfall in ECHOES matches ground truth, so it is a
        # detection, not an unverified claim: the count must not rise.
        assert echoes["unverified_detection_claims"] == silent["unverified_detection_claims"]

    def test_deterministic_warnings_are_detection_claims_not_suggestions(self):
        scored = rb.score(DETERMINISTIC_PREFIX, CASES_1464, config_id="deterministic")
        assert scored["unverified_suggestions"] == 0
        assert scored["unverified_detection_claims"] == 0


# --------------------------------------------------------------------------- #
# Defect 4 — the input hashes are not portable                                 #
# --------------------------------------------------------------------------- #


class TestDefect4PortableHashes:
    def test_crlf_and_lf_hash_identically(self, tmp_path):
        crlf = tmp_path / "crlf.csv"
        lf = tmp_path / "lf.csv"
        crlf.write_bytes(b"a,b\r\n1,2\r\n3,4\r\n")
        lf.write_bytes(b"a,b\n1,2\n3,4\n")
        assert rb.normalized_sha256(crlf) == rb.normalized_sha256(lf)

    def test_a_real_content_change_still_changes_the_hash(self, tmp_path):
        one = tmp_path / "one.csv"
        two = tmp_path / "two.csv"
        one.write_bytes(b"a,b\n1,2\n")
        two.write_bytes(b"a,b\n1,3\n")
        assert rb.normalized_sha256(one) != rb.normalized_sha256(two)

    def test_registry_hashes_are_the_normalised_ones(self):
        if not (BENCH / "data").exists():
            pytest.skip("benchmark data not present in this checkout")
        gt = rb._load_ground_truth()
        assert gt["hash_basis"].startswith("sha256 of the file with CRLF")
        assert rb._verify_inputs(gt) == []

    def test_rescoring_the_same_id_twice_is_refused(self, tmp_path, monkeypatch):
        """A second rescore under the same id would double every row.

        `results.csv` is the evidence ledger, and appending is the only safe
        write to it — which is exactly what makes a repeated `--rescore` a
        quiet way to corrupt it. RUNS is redirected at an empty directory so
        that a broken guard cannot write into the real run records.
        """
        ledger = tmp_path / "results.csv"
        ledger.write_text(
            "run_id,experiment_id\nr1,exp-a-rescored\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(rb, "RESULTS", ledger)
        monkeypatch.setattr(rb, "RUNS", tmp_path / "runs")
        assert rb.experiment_already_recorded("exp-a-rescored") is True
        assert rb.experiment_already_recorded("exp-a") is False
        with pytest.raises(ValueError, match="already"):
            rb.rescore("exp-a", {"datasets": [], "registry_version": "1.1"}, "exp-a-rescored")

    def test_gitattributes_pins_the_benchmark_csvs(self):
        attrs = REPO / ".gitattributes"
        assert attrs.exists(), "no .gitattributes; line endings stay platform-dependent"
        text = attrs.read_text(encoding="utf-8")
        assert "benchmark/data/*.csv" in text
        assert "eol=lf" in text
