"""CLI integration tests for ``mlcompass monitor``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import mlcompass.cli as cli_module
from mlcompass.cli import cli

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def stable_csv_pair(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    ref = tmp_path / "ref.csv"
    cur = tmp_path / "cur.csv"
    pd.DataFrame({"x": rng.normal(0, 1, 500)}).to_csv(ref, index=False)
    pd.DataFrame({"x": rng.normal(0, 1, 500)}).to_csv(cur, index=False)
    return ref, cur


@pytest.fixture
def drifted_csv_pair(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(1)
    ref = tmp_path / "ref.csv"
    cur = tmp_path / "cur.csv"
    pd.DataFrame({"x": rng.normal(0, 1, 500)}).to_csv(ref, index=False)
    pd.DataFrame({"x": rng.normal(3, 1.5, 500)}).to_csv(cur, index=False)
    return ref, cur


# --------------------------------------------------------------------------- #
# Behaviour                                                                   #
# --------------------------------------------------------------------------- #


def test_monitor_stable_run_exits_zero(stable_csv_pair: tuple[Path, Path]) -> None:
    ref, cur = stable_csv_pair
    runner = CliRunner()
    result = runner.invoke(cli, ["monitor", str(ref), str(cur)])
    assert result.exit_code == 0, result.output
    assert "stable" in result.output.lower()


def test_monitor_major_drift_exits_one(drifted_csv_pair: tuple[Path, Path]) -> None:
    """Per the CLI contract, major drift returns exit 1 so CI can gate on it."""
    ref, cur = drifted_csv_pair
    runner = CliRunner()
    result = runner.invoke(cli, ["monitor", str(ref), str(cur)])
    assert result.exit_code == 1
    assert "major_drift" in result.output.lower() or "major drift" in result.output.lower()


def test_monitor_features_filter_passes_through(drifted_csv_pair: tuple[Path, Path]) -> None:
    ref, cur = drifted_csv_pair
    runner = CliRunner()
    # Single-feature filter ⇒ command still runs without crashing.
    result = runner.invoke(cli, ["monitor", str(ref), str(cur), "--features", "x"])
    assert result.exit_code in {0, 1}, result.output
    assert "x" in result.output


def test_monitor_bins_option_respected(stable_csv_pair: tuple[Path, Path]) -> None:
    ref, cur = stable_csv_pair
    runner = CliRunner()
    result = runner.invoke(cli, ["monitor", str(ref), str(cur), "--bins", "5"])
    assert result.exit_code == 0, result.output


def test_monitor_no_overlap_exits_2(tmp_path: Path) -> None:
    ref = tmp_path / "ref.csv"
    cur = tmp_path / "cur.csv"
    pd.DataFrame({"alpha": [1, 2, 3]}).to_csv(ref, index=False)
    pd.DataFrame({"beta": [4, 5, 6]}).to_csv(cur, index=False)
    runner = CliRunner()
    result = runner.invoke(cli, ["monitor", str(ref), str(cur)])
    assert result.exit_code == 2
    assert "no overlapping columns" in result.output.lower()


def test_monitor_llm_skipped_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    stable_csv_pair: tuple[Path, Path],
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ref, cur = stable_csv_pair
    runner = CliRunner()
    result = runner.invoke(cli, ["monitor", str(ref), str(cur), "--llm"])
    assert result.exit_code == 0, result.output
    assert "ANTHROPIC_API_KEY" in result.output


def test_monitor_llm_interpreter_renders_when_callable_returns(
    monkeypatch: pytest.MonkeyPatch,
    stable_csv_pair: tuple[Path, Path],
) -> None:
    """Stub the interpreter so we don't hit the real model."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

    def _stub(_result: dict[str, Any], *, model: str) -> dict[str, Any]:
        return {
            "headline": "Distributions look stable across the board.",
            "likely_cause": "No meaningful change in upstream sources.",
            "next_steps": ["Continue routine monitoring.", "Re-check next week."],
        }

    monkeypatch.setattr(cli_module, "_monitor_interpreter_callable", _stub)

    ref, cur = stable_csv_pair
    runner = CliRunner()
    result = runner.invoke(cli, ["monitor", str(ref), str(cur), "--llm"])
    assert result.exit_code == 0, result.output
    assert "Drift interpretation" in result.output
    assert "Continue routine monitoring" in result.output


def test_root_help_lists_monitor() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "monitor" in result.output
