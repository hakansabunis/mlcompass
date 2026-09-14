"""Tests for ``ProjectContext``."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mlcompass.context import (
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
    assert "mlcompass_version" in meta


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


# --------------------------------------------------------------------------- #
# Concurrent writes                                                           #
# --------------------------------------------------------------------------- #
#
# ``append_decision`` and ``write_context`` are read-modify-write cycles
# over a whole file. Without serialisation two writers interleave as
# read-A / read-B / write-A / write-B and B silently discards A's entry.
# Without atomicity a reader can observe a half-written file, because
# ``write_text`` truncates before it fills.


def test_concurrent_appends_all_survive(tmp_path: Path) -> None:
    """80 threads appending 1 decision each must leave 80 decisions."""
    ctx = ProjectContext.init("race", parent_dir=tmp_path)

    workers = 80
    start = threading.Barrier(workers)

    def _append(i: int) -> None:
        start.wait()
        ctx.append_decision(command="advise", summary=f"decision-{i}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_append, range(workers)))

    decisions = ctx.read_context()["decisions"]
    summaries = {d["summary"] for d in decisions}
    missing = sorted({f"decision-{i}" for i in range(workers)} - summaries)
    assert len(decisions) == workers, f"{workers - len(decisions)} appends lost; missing={missing}"


def test_concurrent_reader_never_observes_a_torn_file(tmp_path: Path) -> None:
    """A reader polling context.json during writes must never see bad JSON.

    ``write_text`` truncates the file and then writes, so any read landing
    in that window gets a prefix of the new content — invalid JSON, or
    valid JSON missing entries. The fix is write-temp-then-replace, which
    makes every observable state either the old file or the new one.
    """
    ctx = ProjectContext.init("torn", parent_dir=tmp_path)
    target = ctx.path / "context.json"

    stop = threading.Event()
    torn: list[str] = []
    reads = [0]

    def _read() -> None:
        while not stop.is_set():
            try:
                raw = target.read_text(encoding="utf-8")
            except OSError:
                # A transient sharing conflict during os.replace is not a
                # torn read — the file was never observed in a bad state.
                continue
            reads[0] += 1
            try:
                payload = json.loads(raw)
            except ValueError:
                torn.append(raw[:200])
                continue
            if not isinstance(payload.get("decisions"), list):
                torn.append(raw[:200])

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    try:
        for i in range(120):
            ctx.append_decision(command="advise", summary=f"d-{i}")
    finally:
        stop.set()
        reader.join(timeout=5)

    assert reads[0] > 0, "reader never managed a read; the test proved nothing"
    assert torn == [], f"reader observed {len(torn)} torn file(s); first: {torn[0]!r}"
