"""Tests for the measurement harness's Phase-0 upgrades.

Focus: the money-safety features (provider wiring, cost estimation, crash-safe
incremental logging with resume) and the R6 telemetry fields. No live calls —
everything runs against the module's pure functions and fake clients.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reproduce_hallucination_ablation.py"

_spec = importlib.util.spec_from_file_location("ablation_harness", SCRIPT)
assert _spec is not None and _spec.loader is not None
harness = importlib.util.module_from_spec(_spec)
sys.modules["ablation_harness"] = harness  # dataclass decorator needs this
_spec.loader.exec_module(harness)

sys.path.insert(0, str(ROOT / "src"))
from mlcompass.agents.leakage_investigator import (  # noqa: E402
    investigate_leakage_bound,
)

# --------------------------------------------------------------------------- #
# Provider table + pricing — the procurement wiring                           #
# --------------------------------------------------------------------------- #

EXPECTED_PROVIDERS = {
    "anthropic",
    "deepseek",
    "openai",
    "gemini",
    "mistral",
    "xai",
    "qwen",
    "groq",
    "vllm",
}


def test_provider_table_covers_the_q1_panel() -> None:
    assert set(harness.PROVIDERS) == EXPECTED_PROVIDERS
    for name, cfg in harness.PROVIDERS.items():
        assert cfg["kind"] in ("openai", "anthropic"), name
        assert cfg["key_env"].endswith("_API_KEY"), name
        assert cfg["model"], name


def test_deepseek_pins_the_direct_model_name() -> None:
    # The 'deepseek-chat' alias is deprecated 2026-07-24; the default must be
    # the direct name so replication runs keep working after that date.
    assert harness.PROVIDERS["deepseek"]["model"] == "deepseek-v4-flash"


def test_every_default_subject_model_has_pricing() -> None:
    # Money-safety: --dry-run must never say "UNKNOWN" for a default model.
    for name, cfg in harness.PROVIDERS.items():
        assert cfg["model"] in harness.PRICING_PER_M, (
            f"default model for provider '{name}' missing from PRICING_PER_M"
        )


def test_estimate_cost_math() -> None:
    usd, calls = harness.estimate_cost_usd("qwen-flash", ("layer1",), 100)
    assert calls == 100
    expected = 100 * (1300 * 0.05 + 300 * 0.40) / 1_000_000.0
    assert usd == pytest.approx(expected)
    # Contract arms include the retry factor.
    _, stress_calls = harness.estimate_cost_usd("qwen-flash", ("layer3_stress",), 100)
    assert stress_calls == 145
    unknown, _ = harness.estimate_cost_usd("no-such-model", ("layer1",), 10)
    assert unknown is None


# --------------------------------------------------------------------------- #
# score_one — the shared per-response scorer                                  #
# --------------------------------------------------------------------------- #


def _resp(**kw: Any) -> dict[str, Any]:
    base = {"columns": [], "claims": [], "verdict": "leakage_likely", "omitted": None}
    base.update(kw)
    return base


def test_score_one_flags_each_channel() -> None:
    allowed = {"a", "anchor_col"}
    corr = {"a": 0.5, "anchor_col": 0.99}
    anchor = "anchor_col"
    ok = _resp(columns=["anchor_col"], claims=[{"column": "a", "statistic": "c", "value": 0.5}])
    assert harness.score_one(ok, allowed, corr, anchor) == {
        "entity": False,
        "value": False,
        "omission": False,
    }
    phantom = _resp(columns=["ghost", "anchor_col"])
    assert harness.score_one(phantom, allowed, corr, anchor)["entity"] is True
    misquote = _resp(
        columns=["anchor_col"], claims=[{"column": "a", "statistic": "c", "value": 0.4}]
    )
    assert harness.score_one(misquote, allowed, corr, anchor)["value"] is True
    omission = _resp(columns=["a"])
    assert harness.score_one(omission, allowed, corr, anchor)["omission"] is True
    abstain = _resp(columns=["a"], verdict="cannot_determine")
    assert harness.score_one(abstain, allowed, corr, anchor)["omission"] is False
    # Contract arms carry the authoritative flag.
    flagged = _resp(columns=["anchor_col"], omitted=True)
    assert harness.score_one(flagged, allowed, corr, anchor)["omission"] is True


def test_normalize_carries_the_telemetry_fields() -> None:
    r = harness._normalize({})
    for key in ("rejection_kinds", "usage_in", "usage_out", "latency_ms"):
        assert key in r


# --------------------------------------------------------------------------- #
# RunLog — crash-safe incremental logging + resume                            #
# --------------------------------------------------------------------------- #


def test_runlog_appends_and_resumes(tmp_path: Path) -> None:
    kw = dict(
        provider="deepseek",
        model="deepseek-v4-flash",
        task_label="synthetic",
        arm="layer1",
        n=4,
        seed=0,
    )
    log = harness.RunLog(str(tmp_path), resume=False, **kw)
    log.append({"i": 0, "columns": ["a"]})
    log.append({"i": 1, "columns": ["b"]})
    # A fresh non-resume open on the same cell must refuse to clobber money.
    with pytest.raises(SystemExit):
        harness.RunLog(str(tmp_path), resume=False, **kw)
    resumed = harness.RunLog(str(tmp_path), resume=True, **kw)
    assert [r["i"] for r in resumed.prior] == [0, 1]


def test_runlog_resume_survives_truncated_tail(tmp_path: Path) -> None:
    kw = dict(
        provider="qwen",
        model="qwen-flash",
        task_label="synthetic",
        arm="layer1",
        n=4,
        seed=0,
    )
    log = harness.RunLog(str(tmp_path), resume=False, **kw)
    log.append({"i": 0})
    with open(log.path, "a", encoding="utf-8") as f:
        f.write('{"i": 1, "columns": [')  # crash mid-write
    resumed = harness.RunLog(str(tmp_path), resume=True, **kw)
    assert [r["i"] for r in resumed.prior] == [0]


def test_runlog_repairs_truncated_tail_before_appending(tmp_path: Path) -> None:
    """The adversarial-review P1: a post-crash append must NOT fuse onto the
    partial line — the repaired log must stay fully parseable so a SECOND
    resume sees every paid record (no silent re-purchase)."""
    kw = dict(
        provider="qwen",
        model="qwen-flash",
        task_label="synthetic",
        arm="layer1",
        n=4,
        seed=0,
    )
    log = harness.RunLog(str(tmp_path), resume=False, **kw)
    log.append({"i": 0})
    with open(log.path, "a", encoding="utf-8") as f:
        f.write('{"i": 1, "columns": [')  # crash mid-write, no newline
    resumed = harness.RunLog(str(tmp_path), resume=True, **kw)
    resumed.append({"i": 1})  # the re-paid record after the crash
    resumed.append({"i": 2})
    second = harness.RunLog(str(tmp_path), resume=True, **kw)
    assert [r["i"] for r in second.prior] == [0, 1, 2]  # nothing lost, no dupes


def test_runlog_filename_separates_different_evidence(tmp_path: Path) -> None:
    """The adversarial-review P2: same basename, different evidence content
    must land in different cells (no pooling of incompatible paid data)."""
    kw = dict(
        provider="qwen",
        model="qwen-flash",
        task_label="csv-insurance.csv-charges",
        arm="layer1",
        n=4,
        seed=0,
    )
    a = harness.RunLog(str(tmp_path), resume=False, evidence_hash="aaaa1111", **kw)
    b = harness.RunLog(str(tmp_path), resume=False, evidence_hash="bbbb2222", **kw)
    assert a.path != b.path


def test_evidence_hash_tracks_content() -> None:
    e1 = {"candidate_leak_columns": ["a"], "target_feature_correlations": []}
    e2 = {"candidate_leak_columns": ["b"], "target_feature_correlations": []}
    assert harness._evidence_hash(e1) != harness._evidence_hash(e2)
    assert harness._evidence_hash(e1) == harness._evidence_hash(dict(e1))


class _RaisingMessages:
    def create(self, **_: Any) -> Any:
        raise RuntimeError("simulated 429/timeout")


class _RaisingAnthropic:
    messages = _RaisingMessages()


def test_contract_arm_survives_transient_provider_error() -> None:
    """The adversarial-review P3: one transient API error on a contract arm
    must yield an empty response (like the open arms), not kill the battery."""
    allowed = ["leak_col", "other_col"]
    out = harness._live_one_response(
        "anthropic", _RaisingAnthropic(), "m", "layer3", EVIDENCE, allowed
    )
    assert out["columns"] == [] and out["claims"] == []
    out2 = harness._live_one_response(
        "anthropic", _RaisingAnthropic(), "m", "stress_mech", EVIDENCE, allowed
    )
    assert out2["columns"] == []


def test_run_cell_resumes_without_new_calls(tmp_path: Path) -> None:
    kw = dict(
        provider="groq",
        model="llama-3.1-8b-instant",
        task_label="synthetic",
        arm="layer1",
        n=2,
        seed=0,
    )
    log = harness.RunLog(str(tmp_path), resume=False, **kw)
    log.append(harness._normalize({}))
    log.append(harness._normalize({}))
    resumed = harness.RunLog(str(tmp_path), resume=True, **kw)
    calls = {"n": 0}

    def one() -> dict[str, Any]:
        calls["n"] += 1
        return harness._normalize({})

    out = harness._run_cell("llama-3.1-8b-instant", 2, (set(), {}, None), resumed, "layer1", one)
    assert len(out) == 2
    assert calls["n"] == 0  # fully resumed — zero paid calls


# --------------------------------------------------------------------------- #
# Zero-cost CLI paths: --dry-run / --check-providers                          #
# --------------------------------------------------------------------------- #


def _run_cli(*argv: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("GEMINI_API_KEY", None)  # prove no key is needed
    return subprocess.run(
        [sys.executable, "-X", "utf8", str(SCRIPT), *argv],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )


def test_dry_run_needs_no_key_and_makes_no_calls() -> None:
    p = _run_cli("--mode", "live", "--provider", "gemini", "--n", "200", "--dry-run")
    assert p.returncode == 0, p.stderr
    assert "ESTIMATED cost" in p.stderr
    assert "no API calls" in p.stderr


def test_check_providers_lists_the_panel() -> None:
    p = _run_cli("--check-providers")
    assert p.returncode == 0, p.stderr
    for name in EXPECTED_PROVIDERS:
        assert name in p.stdout
    assert "No API calls were made" in p.stdout


def test_dry_run_requires_live_mode() -> None:
    p = _run_cli("--dry-run")  # default mode is mock — a silent no-op before the fix
    assert p.returncode != 0
    assert "--mode live" in (p.stderr + p.stdout)


def test_smoke_refuses_resume() -> None:
    p = _run_cli("--mode", "live", "--provider", "gemini", "--smoke", "--resume")
    assert p.returncode != 0
    assert "stale log" in (p.stderr + p.stdout)


def test_max_cost_guard_aborts_before_spending() -> None:
    p = _run_cli(
        "--mode",
        "live",
        "--provider",
        "gemini",
        "--n",
        "200",
        "--max-cost-usd",
        "0.000001",
    )
    assert p.returncode != 0
    assert "exceeds --max-cost-usd" in (p.stderr + p.stdout)


# --------------------------------------------------------------------------- #
# R6 telemetry from the contract path (rejection_kinds / attempts_made)       #
# --------------------------------------------------------------------------- #


class _Block:
    def __init__(self, input_dict: dict[str, Any]) -> None:
        self.type = "tool_use"
        self.name = "submit_investigation"
        self.input = input_dict


class _Resp:
    def __init__(self, input_dict: dict[str, Any]) -> None:
        self.content = [_Block(input_dict)]


class _FakeMessages:
    def __init__(self, scripted: list[dict[str, Any]]) -> None:
        self._scripted = list(scripted)

    def create(self, **_: Any) -> _Resp:
        return _Resp(self._scripted.pop(0))


class _FakeAnthropic:
    def __init__(self, scripted: list[dict[str, Any]]) -> None:
        self.messages = _FakeMessages(scripted)


EVIDENCE = {
    "row_count": 100,
    "trustworthy_sample_size": True,
    "suspicious_metric": {"name": "r2", "value": 1.0},
    "target_feature_correlations": [
        {"feature": "leak_col", "correlation": 0.99},
        {"feature": "other_col", "correlation": 0.42},
    ],
    "perfect_match_rate": 0.99,
    "candidate_leak_columns": ["leak_col"],
    "notes": [],
}


def test_contract_reports_rejection_kinds_and_attempts() -> None:
    good = {
        "verdict": "leakage_likely",
        "confidence": "high",
        "columns_referenced": ["leak_col"],
        "claims": [{"column": "leak_col", "statistic": "correlation", "value": 0.99}],
        "narration": "ok",
    }
    phantom_then_good = [
        {**good, "columns_referenced": ["ghost_col", "leak_col"]},
        good,
    ]
    result = investigate_leakage_bound(
        EVIDENCE, client=_FakeAnthropic(phantom_then_good), model="m"
    )
    assert result["schema_rejections"] == 1
    assert result["rejection_kinds"] == ["entity"]
    assert result["attempts_made"] == 2

    clean = investigate_leakage_bound(EVIDENCE, client=_FakeAnthropic([good]), model="m")
    assert clean["rejection_kinds"] == []
    assert clean["attempts_made"] == 1


# --------------------------------------------------------------------------- #
# FabBench injectors — every pattern must be detectable by the SHIPPED        #
# detector at its thresholds (no API calls; this is the Phase-2 instrument)   #
# --------------------------------------------------------------------------- #


def _make_csv(tmp_path: Path, seed: int = 0) -> Path:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    n = 400
    y = rng.normal(200.0, 40.0, n)
    df = pd.DataFrame(
        {
            "charges": y,
            "age": rng.integers(18, 65, n),
            "bmi": rng.normal(28, 5, n),
            "children": rng.integers(0, 4, n),
        }
    )
    path = tmp_path / "mini_insurance.csv"
    df.to_csv(path, index=False)
    return path


COLUMN_INJECTORS = {
    "exact_copy": "charges_copy_leak",
    "noisy_proxy": "charges_proxy_leak",
    "monotone_log": "log_charges_leak",
    "inverse_target": "inv_charges_leak",
    "binned_target": "charges_enc_leak",
}


@pytest.mark.parametrize("injector,anchor", sorted(COLUMN_INJECTORS.items()))
def test_column_injectors_are_detected_as_top_candidate(
    tmp_path: Path, injector: str, anchor: str
) -> None:
    from mlcompass.agents.leakage_investigator import top_candidate

    csv = _make_csv(tmp_path)
    evidence = harness.build_csv_evidence(str(csv), "charges", seed=0, injector=injector)
    assert top_candidate(evidence) == anchor, (
        f"{injector}: detector did not rank the planted column first "
        f"(candidates: {evidence.get('candidate_leak_columns')})"
    )


def test_contamination_leaks_through_perfect_match_channel(tmp_path: Path) -> None:
    from mlcompass.agents.leakage_investigator import top_candidate

    csv = _make_csv(tmp_path)
    evidence = harness.build_csv_evidence(str(csv), "charges", seed=0, injector="contamination")
    assert evidence["perfect_match_rate"] >= 0.95  # committable under contract rule 4
    assert top_candidate(evidence) is None  # anchor-free evidence shape


def test_injector_registry_matches_roadmap() -> None:
    from fabbench_injectors import list_injectors

    assert list_injectors() == sorted(
        [
            "exact_copy",
            "noisy_proxy",
            "monotone_log",
            "inverse_target",
            "binned_target",
            "contamination",
        ]
    )


def test_unknown_injector_fails_closed() -> None:
    import numpy as np
    import pandas as pd
    from fabbench_injectors import inject

    with pytest.raises(SystemExit):
        inject(pd.DataFrame(), np.array([]), "y", "no_such_pattern", np.random.default_rng(0))


def test_default_injector_reproduces_the_june_task(tmp_path: Path) -> None:
    """Back-compat: the pre-Phase-2 CSV task (log leak) must be byte-identical."""
    csv = _make_csv(tmp_path)
    default = harness.build_csv_evidence(str(csv), "charges", seed=0)
    explicit = harness.build_csv_evidence(str(csv), "charges", seed=0, injector="monotone_log")
    assert harness._evidence_hash(default) == harness._evidence_hash(explicit)


DATA_DIR = ROOT / "scripts" / "data"


@pytest.mark.skipif(
    not (DATA_DIR / "insurance.csv").exists(),
    reason="FabBench datasets not fetched (run scripts/fetch_fabbench_datasets.py)",
)
def test_frozen_phase2_instances_all_verify() -> None:
    """Amendment A1 lock: every frozen (dataset x injector) instance must stay
    detectable by the shipped detector — the zero-cost gate before any paid
    Phase-2 run."""
    spec = importlib.util.spec_from_file_location(
        "fabbench_fetch", ROOT / "scripts" / "fetch_fabbench_datasets.py"
    )
    assert spec is not None and spec.loader is not None
    fetch_mod = importlib.util.module_from_spec(spec)
    sys.modules["fabbench_fetch"] = fetch_mod
    spec.loader.exec_module(fetch_mod)
    assert fetch_mod.verify() == 0


def test_contract_result_normalizer_carries_kinds() -> None:
    out = harness._from_contract_result(
        {
            "columns_referenced": ["leak_col"],
            "claims": [],
            "verdict": "leakage_likely",
            "omitted_critical_evidence": False,
            "schema_rejections": 2,
            "rejection_kinds": ["entity", "entity+value"],
        }
    )
    assert out["rejections"] == 2
    assert out["rejection_kinds"] == ["entity", "entity+value"]
    assert json.dumps(out)  # log-serializable
