"""CLI tests for the ``--llm`` opt-in flag on audit, watch, and compare."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from mlcompass import cli as cli_module
from mlcompass.agents.audit import AuditAgentError
from mlcompass.agents.compare import CompareAgentError
from mlcompass.agents.watch import WatchAgentError
from mlcompass.cli import cli
from mlcompass.context import ProjectContext

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")


@pytest.fixture
def without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def buggy_script(tmp_path: Path) -> Path:
    code = """
import torch
import torch.nn as nn
model = nn.LSTM(10, 20)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, momentum=0.9)
"""
    p = tmp_path / "train.py"
    p.write_text(code, encoding="utf-8")
    return p


@pytest.fixture
def nan_log(tmp_path: Path) -> Path:
    lines = [f"Epoch {i} train_loss={0.8 - i * 0.1:.4f}" for i in range(6)] + [
        "Epoch 6 train_loss=nan"
    ]
    p = tmp_path / "train.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


@pytest.fixture
def two_runs(tmp_path: Path) -> tuple[Path, Path]:
    def _make(run_id: str, cfg: dict, metrics: list[dict]) -> Path:
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump({"name": run_id, "config": cfg}),
            encoding="utf-8",
        )
        (run_dir / "metrics.json").write_text(json.dumps({"metrics": metrics}), encoding="utf-8")
        return run_dir

    a = _make("run-3", {"lr": 1e-3}, [{"epoch": 0, "val_loss": 0.5}])
    b = _make("run-7", {"lr": 3e-4}, [{"epoch": 0, "val_loss": 0.3}])
    return a, b


# --------------------------------------------------------------------------- #
# audit --llm                                                                 #
# --------------------------------------------------------------------------- #


def test_audit_llm_calls_prioritizer(
    buggy_script: Path,
    tmp_path: Path,
    with_api_key: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"called": False, "model": None}

    def fake(result: dict[str, Any], *, model: str = "claude-opus-4-7") -> dict[str, Any]:
        captured["called"] = True
        captured["model"] = model
        return {
            "priorities": [
                {
                    "rule_id": "seed",
                    "priority_rank": 1,
                    "blast_radius": "results irreproducible.",
                }
            ],
            "synthesis": "Fix the seed first.",
        }

    monkeypatch.setattr(cli_module, "_audit_prioritizer_callable", fake)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["audit", str(buggy_script), "--llm"])

    assert result.exit_code == 0, result.output
    assert captured["called"]
    # Output should include the prioritizer's verdict text.
    assert "fix the seed first" in result.output.lower()


def test_audit_llm_skips_when_no_api_key(
    buggy_script: Path,
    tmp_path: Path,
    without_api_key: None,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["audit", str(buggy_script), "--llm"])
    assert result.exit_code == 0
    assert "anthropic_api_key" in result.output.lower()


def test_audit_llm_skips_when_no_findings(
    tmp_path: Path,
    with_api_key: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean = tmp_path / "clean.py"
    clean.write_text(
        "import torch\n"
        "import torch.nn as nn\n"
        "from sklearn.model_selection import train_test_split\n"
        "torch.manual_seed(42)\n"
        "X_tr, X_v, y_tr, y_v = train_test_split(X, y, test_size=0.2)\n"
        "model = nn.Linear(10, 1)\n"
        "model.train()\n"
        "model.eval()\n"
        "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)\n"
        "loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)\n",
        encoding="utf-8",
    )

    called = {"flag": False}

    def fake(_result: dict[str, Any], *, model: str = "claude-opus-4-7") -> dict[str, Any]:
        called["flag"] = True
        return {"priorities": [], "synthesis": ""}

    monkeypatch.setattr(cli_module, "_audit_prioritizer_callable", fake)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["audit", str(clean), "--llm"])
    assert result.exit_code == 0
    assert not called["flag"], "should short-circuit when no findings"
    assert "nothing to prioritize" in result.output.lower()


def test_audit_llm_handles_agent_error(
    buggy_script: Path,
    tmp_path: Path,
    with_api_key: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AuditAgentError("model returned garbage")

    monkeypatch.setattr(cli_module, "_audit_prioritizer_callable", broken)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["audit", str(buggy_script), "--llm"])
    assert result.exit_code == 0
    assert "bad response" in result.output.lower() or "garbage" in result.output.lower()


def test_audit_llm_persists_priorities_to_project(
    buggy_script: Path,
    tmp_path: Path,
    with_api_key: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectContext.init("test-proj", parent_dir=tmp_path)

    monkeypatch.setattr(
        cli_module,
        "_audit_prioritizer_callable",
        lambda result, *, model="claude-opus-4-7": {
            "priorities": [{"rule_id": "seed", "priority_rank": 1, "blast_radius": "X"}],
            "synthesis": "S",
        },
    )

    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["audit", str(buggy_script), "--llm"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0, result.output
    log_path = project.path / "advice.log"
    entry = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert "priorities" in entry
    assert entry["priorities"]["priorities"][0]["rule_id"] == "seed"


# --------------------------------------------------------------------------- #
# watch --llm                                                                 #
# --------------------------------------------------------------------------- #


def test_watch_llm_calls_diagnostician(
    nan_log: Path,
    tmp_path: Path,
    with_api_key: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"snapshots": None, "findings": None}

    def fake(
        snapshots: list[dict], findings: list[dict], *, model: str = "claude-opus-4-7"
    ) -> dict:
        captured["snapshots"] = snapshots
        captured["findings"] = findings
        return {
            "diagnosis": [
                {
                    "finding_rule_id": "nan",
                    "hypothesis": "log(0) most likely.",
                    "recommended_action": "Add epsilon to loss.",
                    "confidence": "high",
                }
            ],
            "summary": "Stop training, fix loss formula.",
        }

    monkeypatch.setattr(cli_module, "_watch_diagnostician_callable", fake)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["watch", str(nan_log), "--llm"])
    assert result.exit_code == 0, result.output
    assert captured["snapshots"] is not None
    assert any(f["rule_id"] == "nan" for f in captured["findings"])
    assert "stop training" in result.output.lower()


def test_watch_llm_skips_when_no_findings(
    tmp_path: Path,
    with_api_key: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = tmp_path / "clean.log"
    log.write_text(
        "\n".join(f"Epoch {i} train_loss={0.8 - i * 0.08:.3f}" for i in range(8)),
        encoding="utf-8",
    )
    called = {"flag": False}

    def fake(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        called["flag"] = True
        return {"diagnosis": [], "summary": ""}

    monkeypatch.setattr(cli_module, "_watch_diagnostician_callable", fake)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["watch", str(log), "--llm"])
    assert result.exit_code == 0
    assert not called["flag"]
    assert "nothing to diagnose" in result.output.lower()


def test_watch_llm_skips_when_no_api_key(
    nan_log: Path,
    tmp_path: Path,
    without_api_key: None,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["watch", str(nan_log), "--llm"])
    assert result.exit_code == 0
    assert "anthropic_api_key" in result.output.lower()


def test_watch_llm_handles_agent_error(
    nan_log: Path,
    tmp_path: Path,
    with_api_key: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_a: Any, **_kw: Any) -> dict[str, Any]:
        raise WatchAgentError("malformed")

    monkeypatch.setattr(cli_module, "_watch_diagnostician_callable", broken)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["watch", str(nan_log), "--llm"])
    assert result.exit_code == 0
    assert "bad response" in result.output.lower() or "malformed" in result.output.lower()


# --------------------------------------------------------------------------- #
# compare --llm                                                               #
# --------------------------------------------------------------------------- #


def test_compare_llm_calls_hypothesizer(
    two_runs: tuple[Path, Path],
    tmp_path: Path,
    with_api_key: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a, b = two_runs
    captured: dict[str, Any] = {"called": False}

    def fake(comparison: dict[str, Any], *, model: str = "claude-opus-4-7") -> dict[str, Any]:
        captured["called"] = True
        return {
            "hypothesis": "B's lower lr won.",
            "key_factors": [{"config_key": "lr", "impact": "high", "reason": "Only change."}],
            "next_experiment": "Try lr=1e-4 on Run B's setup.",
        }

    monkeypatch.setattr(cli_module, "_compare_hypothesizer_callable", fake)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["compare", str(a), str(b), "--llm"])
    assert result.exit_code == 0, result.output
    assert captured["called"]
    assert "lower lr won" in result.output.lower()


def test_compare_llm_skips_when_no_api_key(
    two_runs: tuple[Path, Path],
    tmp_path: Path,
    without_api_key: None,
) -> None:
    a, b = two_runs
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["compare", str(a), str(b), "--llm"])
    assert result.exit_code == 0
    assert "anthropic_api_key" in result.output.lower()


def test_compare_llm_handles_agent_error(
    two_runs: tuple[Path, Path],
    tmp_path: Path,
    with_api_key: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a, b = two_runs

    def broken(*_a: Any, **_kw: Any) -> dict[str, Any]:
        raise CompareAgentError("bad shape")

    monkeypatch.setattr(cli_module, "_compare_hypothesizer_callable", broken)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["compare", str(a), str(b), "--llm"])
    assert result.exit_code == 0
    assert "bad response" in result.output.lower() or "bad shape" in result.output.lower()
