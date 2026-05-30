"""Tests for ``tools.runs`` (load_run, compare_runs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mlcompass.context import ProjectContext
from mlcompass.tools.runs import (
    RunNotFoundError,
    RunRecord,
    compare_runs,
    load_run,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _make_run(
    parent: Path,
    *,
    run_id: str,
    config: dict | None = None,
    metrics: list[dict] | None = None,
    name: str | None = None,
    notes: str | None = None,
) -> Path:
    """Build a run directory at ``parent/run_id``."""
    run_dir = parent / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta: dict = {}
    if name is not None:
        meta["name"] = name
    meta["created"] = "2026-05-29T15:00:00Z"
    meta["config"] = config or {}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")

    metric_payload = {"metrics": metrics or []}
    (run_dir / "metrics.json").write_text(json.dumps(metric_payload), encoding="utf-8")

    if notes is not None:
        (run_dir / "notes.md").write_text(notes, encoding="utf-8")

    return run_dir


# --------------------------------------------------------------------------- #
# load_run                                                                    #
# --------------------------------------------------------------------------- #


def test_load_run_by_direct_path(tmp_path: Path) -> None:
    run_dir = _make_run(
        tmp_path,
        run_id="run-001",
        name="baseline",
        config={"lr": 1e-3, "batch_size": 64},
        metrics=[{"epoch": 0, "train_loss": 0.8, "val_loss": 0.75}],
    )

    record = load_run(run_dir)

    assert isinstance(record, RunRecord)
    assert record.id == "run-001"
    assert record.name == "baseline"
    assert record.config == {"lr": 1e-3, "batch_size": 64}
    assert len(record.metrics) == 1
    assert record.final_metrics["val_loss"] == 0.75


def test_load_run_by_id_inside_project(tmp_path: Path) -> None:
    project = ProjectContext.init("test-proj", parent_dir=tmp_path)
    _make_run(
        project.path / "runs",
        run_id="run-007",
        config={"lr": 3e-4},
        metrics=[{"epoch": 0, "val_loss": 0.5}],
    )

    record = load_run("run-007", project=project)

    assert record.id == "run-007"
    assert record.config["lr"] == 3e-4


def test_load_run_raises_when_missing(tmp_path: Path) -> None:
    project = ProjectContext.init("test-proj", parent_dir=tmp_path)
    with pytest.raises(RunNotFoundError):
        load_run("nope", project=project)


def test_load_run_accepts_metrics_as_bare_list(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-x"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("config: {}\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps([{"epoch": 0, "val_loss": 0.4}]),
        encoding="utf-8",
    )

    record = load_run(run_dir)
    assert record.metrics == [{"epoch": 0, "val_loss": 0.4}]


def test_load_run_reads_notes(tmp_path: Path) -> None:
    run_dir = _make_run(
        tmp_path,
        run_id="run-n",
        config={},
        notes="First baseline experiment\n",
    )
    record = load_run(run_dir)
    assert record.notes is not None
    assert "baseline" in record.notes


def test_final_metrics_handles_empty(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, run_id="run-empty", config={}, metrics=[])
    record = load_run(run_dir)
    assert record.final_metrics == {}


# --------------------------------------------------------------------------- #
# compare_runs — config_diff                                                  #
# --------------------------------------------------------------------------- #


def _record(
    tmp_path: Path,
    run_id: str,
    config: dict,
    metrics: list[dict],
) -> RunRecord:
    return load_run(_make_run(tmp_path, run_id=run_id, config=config, metrics=metrics))


def test_compare_returns_top_level_keys(tmp_path: Path) -> None:
    a = _record(tmp_path, "a", {"lr": 1e-3}, [{"epoch": 0, "val_loss": 0.5}])
    b = _record(tmp_path, "b", {"lr": 1e-3}, [{"epoch": 0, "val_loss": 0.5}])
    result = compare_runs(a, b)
    for key in (
        "run_a",
        "run_b",
        "config_diff",
        "metric_comparison",
        "verdict",
        "verdict_explanation",
    ):
        assert key in result


def test_config_diff_reports_changed_keys(tmp_path: Path) -> None:
    a = _record(tmp_path, "a", {"lr": 1e-3, "batch_size": 64}, [])
    b = _record(tmp_path, "b", {"lr": 3e-4, "batch_size": 64}, [])
    result = compare_runs(a, b)
    diff_keys = [d["key"] for d in result["config_diff"]]
    assert "lr" in diff_keys
    assert "batch_size" not in diff_keys


def test_config_diff_includes_missing_keys(tmp_path: Path) -> None:
    a = _record(tmp_path, "a", {"lr": 1e-3}, [])
    b = _record(tmp_path, "b", {"lr": 1e-3, "dropout": 0.3}, [])
    result = compare_runs(a, b)
    diff = {d["key"]: d for d in result["config_diff"]}
    assert "dropout" in diff
    assert diff["dropout"]["a_value"] is None
    assert diff["dropout"]["b_value"] == 0.3


# --------------------------------------------------------------------------- #
# compare_runs — metric direction + verdict                                   #
# --------------------------------------------------------------------------- #


def test_lower_is_better_for_loss(tmp_path: Path) -> None:
    a = _record(tmp_path, "a", {}, [{"epoch": 0, "val_loss": 0.5}])
    b = _record(tmp_path, "b", {}, [{"epoch": 0, "val_loss": 0.3}])
    result = compare_runs(a, b)
    rows = {r["name"]: r for r in result["metric_comparison"]}
    assert rows["val_loss"]["better"] == "b"
    assert result["verdict"] == "b_better"


def test_higher_is_better_for_accuracy(tmp_path: Path) -> None:
    a = _record(tmp_path, "a", {}, [{"epoch": 0, "val_acc": 0.85}])
    b = _record(tmp_path, "b", {}, [{"epoch": 0, "val_acc": 0.78}])
    result = compare_runs(a, b)
    rows = {r["name"]: r for r in result["metric_comparison"]}
    assert rows["val_acc"]["better"] == "a"
    assert result["verdict"] == "a_better"


def test_mixed_verdict_when_metrics_disagree(tmp_path: Path) -> None:
    a = _record(
        tmp_path,
        "a",
        {},
        [{"epoch": 0, "val_loss": 0.3, "val_acc": 0.80}],
    )
    b = _record(
        tmp_path,
        "b",
        {},
        [{"epoch": 0, "val_loss": 0.2, "val_acc": 0.78}],
    )
    result = compare_runs(a, b)
    assert result["verdict"] == "mixed"


def test_inconclusive_when_no_known_direction(tmp_path: Path) -> None:
    a = _record(tmp_path, "a", {}, [{"epoch": 0, "wibble": 1.0}])
    b = _record(tmp_path, "b", {}, [{"epoch": 0, "wibble": 2.0}])
    result = compare_runs(a, b)
    assert result["verdict"] == "inconclusive"


def test_metric_comparison_skips_non_numeric(tmp_path: Path) -> None:
    a = _record(tmp_path, "a", {}, [{"epoch": 0, "val_loss": 0.3, "tag": "good"}])
    b = _record(tmp_path, "b", {}, [{"epoch": 0, "val_loss": 0.4, "tag": "bad"}])
    result = compare_runs(a, b)
    names = [r["name"] for r in result["metric_comparison"]]
    assert "tag" not in names
    assert "val_loss" in names


def test_metric_comparison_skips_epoch_field(tmp_path: Path) -> None:
    a = _record(tmp_path, "a", {}, [{"epoch": 9, "val_loss": 0.3}])
    b = _record(tmp_path, "b", {}, [{"epoch": 20, "val_loss": 0.2}])
    result = compare_runs(a, b)
    names = [r["name"] for r in result["metric_comparison"]]
    assert "epoch" not in names


def test_delta_sign_matches_direction(tmp_path: Path) -> None:
    a = _record(tmp_path, "a", {}, [{"epoch": 0, "val_loss": 0.5}])
    b = _record(tmp_path, "b", {}, [{"epoch": 0, "val_loss": 0.3}])
    result = compare_runs(a, b)
    row = result["metric_comparison"][0]
    # delta = B − A
    assert row["delta"] == pytest.approx(-0.2)


def test_run_summary_has_epoch_count(tmp_path: Path) -> None:
    a = _record(tmp_path, "a", {}, [{"epoch": i, "val_loss": 1.0 / (i + 1)} for i in range(5)])
    b = _record(tmp_path, "b", {}, [{"epoch": 0, "val_loss": 0.4}])
    result = compare_runs(a, b)
    assert result["run_a"]["epochs"] == 5
    assert result["run_b"]["epochs"] == 1
