"""Tests for the HPO sub-agent tool layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mlcompass.tools.optimize import (
    OptimizeAnalysisError,
    _perturb,
    _rank,
    load_runs_from_dir,
    optimize_history,
    parse_constraints,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _make_run(
    runs_dir: Path,
    name: str,
    config: dict,
    metrics: list[dict],
) -> Path:
    run = runs_dir / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(
        yaml.safe_dump({"name": name, "config": config}),
        encoding="utf-8",
    )
    (run / "metrics.json").write_text(json.dumps({"metrics": metrics}), encoding="utf-8")
    return run


@pytest.fixture
def monotone_history(tmp_path: Path) -> Path:
    """lr-strictly-helps history: lower lr ⇒ higher val_acc."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    pairs = [(0.01, 0.62), (0.001, 0.75), (0.0001, 0.86), (0.005, 0.69), (0.0005, 0.81)]
    for i, (lr, acc) in enumerate(pairs):
        _make_run(
            runs_dir,
            f"run-{i}",
            {"lr": lr, "batch_size": 64, "dropout": 0.2},
            [{"epoch": 0, "val_acc": acc - 0.1}, {"epoch": 1, "val_acc": acc}],
        )
    return runs_dir


# --------------------------------------------------------------------------- #
# Numerics                                                                    #
# --------------------------------------------------------------------------- #


def test_rank_handles_ties() -> None:
    import numpy as np

    ranks = _rank(np.array([10.0, 20.0, 20.0, 40.0]))
    # 10→1, the two 20s share rank 2.5, 40→4
    assert list(ranks) == [1.0, 2.5, 2.5, 4.0]


def test_perturb_multiplicative_for_positive_observations() -> None:
    """Strictly-positive observations must NEVER produce a negative result."""
    new = _perturb(
        current=0.0001,
        corr=-1.0,
        direction="max",
        step_frac=0.5,
        obs_min=0.0001,
        obs_max=0.01,
        bounds=None,
    )
    assert new > 0
    # Sign of corr is negative, direction is max ⇒ "exploit" should step DOWN
    # because higher-rank lr (small lr) has higher metric.
    assert new < 0.0001


def test_perturb_additive_when_bounds_supplied() -> None:
    """User bounds force additive scheme — sign can cross zero if user says so."""
    new = _perturb(
        current=0.5,
        corr=1.0,
        direction="max",
        step_frac=0.5,
        obs_min=0.1,
        obs_max=1.0,
        bounds=(-2.0, 2.0),
    )
    assert -2.0 <= new <= 2.0


def test_perturb_integer_stays_integer() -> None:
    new = _perturb(
        current=64,
        corr=0.5,
        direction="max",
        step_frac=0.5,
        obs_min=32,
        obs_max=128,
        bounds=None,
    )
    assert isinstance(new, int)


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #


def test_load_runs_from_dir_skips_non_runs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_run(runs_dir, "good", {"lr": 0.01}, [{"epoch": 0, "val_acc": 0.5}])
    (runs_dir / "noise.txt").write_text("ignore me", encoding="utf-8")
    (runs_dir / "broken").mkdir()  # No config.yaml.

    records = load_runs_from_dir(runs_dir)
    assert [r.id for r in records] == ["good"]


def test_load_runs_from_dir_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(OptimizeAnalysisError, match="not found"):
        load_runs_from_dir(tmp_path / "does-not-exist")


# --------------------------------------------------------------------------- #
# optimize_history                                                            #
# --------------------------------------------------------------------------- #


def test_optimize_history_picks_best_lr_correctly(monotone_history: Path) -> None:
    runs = load_runs_from_dir(monotone_history)
    result = optimize_history(runs, metric="val_acc", direction="max")
    assert result["best"]["config"]["lr"] == 0.0001
    assert result["best"]["score"] == 0.86


def test_optimize_history_sensitivity_finds_dominant_knob(
    monotone_history: Path,
) -> None:
    runs = load_runs_from_dir(monotone_history)
    result = optimize_history(runs, metric="val_acc", direction="max")
    # Top sensitivity entry should be lr with strong negative corr
    top = result["sensitivity"][0]
    assert top["hyperparam"] == "lr"
    assert top["correlation"] is not None
    assert top["correlation"] < -0.8


def test_optimize_history_suggestions_are_positive_for_lr(
    monotone_history: Path,
) -> None:
    runs = load_runs_from_dir(monotone_history)
    result = optimize_history(runs, metric="val_acc", direction="max", n_suggestions=3)
    for sug in result["suggestions"]:
        lr = sug["config"]["lr"]
        assert lr > 0, f"Suggested lr {lr} must be positive"


def test_optimize_history_metric_not_found_raises(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_run(runs_dir, "r1", {"lr": 0.01}, [{"epoch": 0, "loss": 1.0}])
    runs = load_runs_from_dir(runs_dir)
    with pytest.raises(OptimizeAnalysisError, match="exposes the metric"):
        optimize_history(runs, metric="val_acc", direction="max")


def test_optimize_history_empty_runs_raises() -> None:
    with pytest.raises(OptimizeAnalysisError, match="No runs"):
        optimize_history([], metric="val_acc", direction="max")


def test_optimize_history_bad_direction_raises(monotone_history: Path) -> None:
    runs = load_runs_from_dir(monotone_history)
    with pytest.raises(OptimizeAnalysisError, match="must be 'max' or 'min'"):
        optimize_history(runs, metric="val_acc", direction="sideways")


def test_optimize_history_warns_when_partial_metric_coverage(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_run(runs_dir, "complete", {"lr": 0.01}, [{"epoch": 0, "val_acc": 0.8}])
    _make_run(runs_dir, "missing", {"lr": 0.001}, [{"epoch": 0, "loss": 0.5}])
    # Two more so we hit the 3-run sensitivity floor.
    _make_run(runs_dir, "extra1", {"lr": 0.005}, [{"epoch": 0, "val_acc": 0.7}])
    _make_run(runs_dir, "extra2", {"lr": 0.0001}, [{"epoch": 0, "val_acc": 0.9}])

    runs = load_runs_from_dir(runs_dir)
    result = optimize_history(runs, metric="val_acc", direction="max")
    assert any("skipped" in w for w in result["warnings"])
    assert result["n_scored"] == 3
    assert result["n_runs"] == 4


def test_optimize_history_min_direction_picks_lowest(monotone_history: Path) -> None:
    """Sanity: with direction='min' the best run should be the WORST val_acc."""
    runs = load_runs_from_dir(monotone_history)
    result = optimize_history(runs, metric="val_acc", direction="min")
    # Worst acc was the lr=0.01 run (val_acc=0.62)
    assert result["best"]["config"]["lr"] == 0.01


# --------------------------------------------------------------------------- #
# parse_constraints                                                           #
# --------------------------------------------------------------------------- #


def test_parse_constraints_basic() -> None:
    out = parse_constraints("lr:0.0001-0.1,batch_size:16-256")
    assert out == {"lr": (0.0001, 0.1), "batch_size": (16.0, 256.0)}


def test_parse_constraints_empty_string_returns_empty() -> None:
    assert parse_constraints("") == {}


def test_parse_constraints_normalises_swapped_bounds() -> None:
    out = parse_constraints("lr:0.1-0.0001")
    assert out["lr"] == (0.0001, 0.1)


def test_parse_constraints_bad_segment_raises() -> None:
    with pytest.raises(OptimizeAnalysisError):
        parse_constraints("lr=0.001")
