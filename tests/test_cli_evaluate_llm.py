"""CLI tests for ``mlcompass evaluate --llm``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from mlcompass import cli as cli_module
from mlcompass.agents.evaluate import EvaluateAgentError
from mlcompass.cli import cli
from mlcompass.context import ProjectContext

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def binary_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 80
    y_true = rng.integers(0, 2, size=n)
    y_prob = np.where(
        y_true == 1,
        rng.uniform(0.55, 0.95, size=n),
        rng.uniform(0.05, 0.45, size=n),
    )
    y_pred = (y_prob >= 0.5).astype(int)
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob})
    p = tmp_path / "preds.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    # v0.7: when the leakage-smell threshold fires (which it does on
    # ``binary_csv`` because the synthesised predictions are very
    # accurate), the CLI also calls the leakage investigator. Stub it
    # so we don't accidentally hit the real Anthropic API mid-test.
    monkeypatch.setattr(
        cli_module,
        "_leakage_investigator_callable",
        lambda evidence, *, model: {
            "verdict": "leakage_uncertain",
            "confidence": "low",
            "evidence_cited": [],
            "primary_hypothesis": "(stub)",
            "recommended_checks": [],
        },
    )


@pytest.fixture
def without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


# --------------------------------------------------------------------------- #
# Happy path                                                                  #
# --------------------------------------------------------------------------- #


def test_evaluate_llm_calls_interpreter(
    binary_csv: Path,
    tmp_path: Path,
    with_api_key: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"called": False, "task": None}

    def fake(evaluation: dict[str, Any], *, model: str = "claude-opus-4-7") -> dict[str, Any]:
        captured["called"] = True
        captured["task"] = evaluation.get("task")
        return {
            "assessment": "Overall strong.",
            "strengths": ["AUC is high."],
            "weaknesses": ["Precision could be tighter."],
            "next_steps": ["Ship threshold 0.65."],
        }

    monkeypatch.setattr(cli_module, "_evaluate_interpreter_callable", fake)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["evaluate", str(binary_csv), "--llm"])

    assert result.exit_code == 0, result.output
    assert captured["called"]
    assert captured["task"] == "binary_classification"
    text = result.output.lower()
    assert "assessment" in text
    assert "ship threshold 0.65" in text


def test_evaluate_llm_skips_when_no_api_key(
    binary_csv: Path,
    tmp_path: Path,
    without_api_key: None,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["evaluate", str(binary_csv), "--llm"])
    assert result.exit_code == 0
    assert "anthropic_api_key" in result.output.lower()


def test_evaluate_llm_handles_agent_error(
    binary_csv: Path,
    tmp_path: Path,
    with_api_key: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_a: Any, **_kw: Any) -> dict[str, Any]:
        raise EvaluateAgentError("model returned garbage")

    monkeypatch.setattr(cli_module, "_evaluate_interpreter_callable", broken)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["evaluate", str(binary_csv), "--llm"])
    assert result.exit_code == 0
    assert "bad response" in result.output.lower() or "garbage" in result.output.lower()


def test_evaluate_llm_persists_interpretation_to_project(
    binary_csv: Path,
    tmp_path: Path,
    with_api_key: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = ProjectContext.init("test-proj", parent_dir=tmp_path)

    def fake(evaluation: dict[str, Any], *, model: str = "claude-opus-4-7") -> dict[str, Any]:
        return {
            "assessment": "Strong.",
            "strengths": ["s1"],
            "weaknesses": ["w1"],
            "next_steps": ["n1"],
        }

    monkeypatch.setattr(cli_module, "_evaluate_interpreter_callable", fake)

    runner = CliRunner()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["evaluate", str(binary_csv), "--llm"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0, result.output

    log_path = project.path / "advice.log"
    entries = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert entries
    entry = entries[0]
    assert "interpretation" in entry
    assert entry["interpretation"]["next_steps"] == ["n1"]
