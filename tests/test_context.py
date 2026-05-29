"""Tests for ``ProjectContext``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_copilot.context import (
    DEFAULT_PROJECT_DIR,
    ProjectContext,
    ProjectExistsError,
    ProjectNotFoundError,
)


def test_init_creates_directory_tree(tmp_path: Path) -> None:
    ctx = ProjectContext.init("test-proj", parent_dir=tmp_path)

    assert ctx.path == tmp_path / DEFAULT_PROJECT_DIR
    assert ctx.path.is_dir()
    assert (ctx.path / "project.yaml").is_file()
    assert (ctx.path / "context.json").is_file()
    assert (ctx.path / "datasets").is_dir()
    assert (ctx.path / "runs").is_dir()
    assert (ctx.path / "cache").is_dir()
    assert (ctx.path / "advice.log").is_file()
    assert (ctx.path / ".gitignore").is_file()


def test_init_writes_project_meta(tmp_path: Path) -> None:
    ctx = ProjectContext.init("test-proj", parent_dir=tmp_path)
    meta = ctx.project_meta

    assert meta["name"] == "test-proj"
    assert meta["default_model"] == "claude-opus-4-7"
    assert "created" in meta
    assert "ml_copilot_version" in meta


def test_init_writes_empty_context(tmp_path: Path) -> None:
    ctx = ProjectContext.init("test-proj", parent_dir=tmp_path)
    state = ctx.read_context()

    assert state["project_type"] is None
    assert state["target_column"] is None
    assert state["preferred_models"] == []
    assert state["active_dataset"] is None
    assert state["current_run"] is None
    assert state["decisions"] == []


def test_init_raises_when_already_exists(tmp_path: Path) -> None:
    ProjectContext.init("test-proj", parent_dir=tmp_path)
    with pytest.raises(ProjectExistsError):
        ProjectContext.init("another", parent_dir=tmp_path)


def test_init_respects_custom_default_model(tmp_path: Path) -> None:
    ctx = ProjectContext.init(
        "test-proj",
        parent_dir=tmp_path,
        default_model="claude-haiku-4-5",
    )
    assert ctx.project_meta["default_model"] == "claude-haiku-4-5"


def test_load_finds_existing(tmp_path: Path) -> None:
    ProjectContext.init("test-proj", parent_dir=tmp_path)
    ctx = ProjectContext.load(search_from=tmp_path)
    assert ctx.path == tmp_path / DEFAULT_PROJECT_DIR


def test_load_walks_up_to_find_project(tmp_path: Path) -> None:
    ProjectContext.init("test-proj", parent_dir=tmp_path)
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    ctx = ProjectContext.load(search_from=deep)
    assert ctx.path == tmp_path / DEFAULT_PROJECT_DIR


def test_load_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(ProjectNotFoundError):
        ProjectContext.load(search_from=tmp_path)


def test_write_context_merges_top_level(tmp_path: Path) -> None:
    ctx = ProjectContext.init("test-proj", parent_dir=tmp_path)
    ctx.write_context({"project_type": "binary_classification"})

    state = ctx.read_context()
    assert state["project_type"] == "binary_classification"
    # Other keys must be preserved.
    assert state["decisions"] == []
    assert state["preferred_models"] == []


def test_append_decision_records_timestamp_and_fields(tmp_path: Path) -> None:
    ctx = ProjectContext.init("test-proj", parent_dir=tmp_path)
    ctx.append_decision(
        command="advise",
        summary="Recommended XGBoost as baseline",
        reasoning="Tabular binary classification, 10K rows",
    )

    state = ctx.read_context()
    assert len(state["decisions"]) == 1

    decision = state["decisions"][0]
    assert decision["command"] == "advise"
    assert decision["summary"] == "Recommended XGBoost as baseline"
    assert decision["reasoning"] == "Tabular binary classification, 10K rows"
    assert "timestamp" in decision


def test_register_dataset_creates_metadata_file(tmp_path: Path) -> None:
    ctx = ProjectContext.init("test-proj", parent_dir=tmp_path)

    fake_csv = tmp_path / "data.csv"
    fake_csv.write_text("a,b\n1,2\n", encoding="utf-8")

    fingerprint = ctx.register_dataset(fake_csv, meta={"rows": 1, "cols": 2})

    assert len(fingerprint) == 16
    meta_file = ctx.path / "datasets" / f"{fingerprint}.json"
    assert meta_file.is_file()

    saved = json.loads(meta_file.read_text(encoding="utf-8"))
    assert saved["rows"] == 1
    assert saved["cols"] == 2
    assert saved["path"] == str(fake_csv.resolve())
    assert "registered_at" in saved
