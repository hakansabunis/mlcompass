"""The ledger write boundary: which project a tool is allowed to write into.

Every mlcompass tool that is declared read-only still appends to a
``.mlcompass/`` ledger (decisions, ``advice.log``, active-state fields,
``datasets/``). That is deliberate — it is the audit trail. What is *not*
deliberate is letting a **path argument** choose which project gets
written. From project A, a single ``advise`` naming a file inside project
B used to write into B.

The rule these tests pin down:

    Ledger writes always target the ACTIVE project — the one found by
    walking up from the agent's / server's own working root. A path
    argument never selects a write target. When a path argument resolves
    to a different project, the write is refused and the tool result
    carries a ``ledger`` note saying so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mlcompass.context import ProjectContext
from mlcompass.mcp_server import (
    mlcompass_advise,
    mlcompass_audit,
    mlcompass_compare,
    mlcompass_deploy,
    mlcompass_evaluate,
    mlcompass_watch,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _snapshot(root: Path) -> dict[str, bytes]:
    """Byte-exact snapshot of every file under ``root``."""
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def _make_run(path: Path, name: str, *, lr: float, val_acc: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text(
        f"name: {name}\nconfig:\n  lr: {lr}\n  batch_size: 64\n",
        encoding="utf-8",
    )
    (path / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": [
                    {"epoch": 0, "train_loss": 0.8, "val_loss": 0.7, "val_acc": 0.6},
                    {"epoch": 1, "train_loss": 0.4, "val_loss": 0.5, "val_acc": val_acc},
                ]
            }
        ),
        encoding="utf-8",
    )


def _populate(root: Path) -> dict[str, Path]:
    """Drop one valid input for each read-only tool under ``root``."""
    csv = root / "data.csv"
    pd.DataFrame({"age": list(range(20)), "churn": [0, 1] * 10}).to_csv(csv, index=False)

    preds = root / "preds.csv"
    pd.DataFrame({"y_true": [0, 1] * 10, "y_pred": [0, 1] * 10}).to_csv(preds, index=False)

    script = root / "train.py"
    script.write_text("import torch\nm = torch.nn.Linear(1, 1)\n", encoding="utf-8")

    log = root / "train.log"
    log.write_text(
        "\n".join(
            [
                "Epoch 0: train_loss=0.80 val_loss=0.75",
                "Epoch 1: train_loss=0.50 val_loss=0.55",
                "Epoch 2: train_loss=0.35 val_loss=0.45",
            ]
        ),
        encoding="utf-8",
    )

    model = root / "model.pt"
    model.write_bytes(b"PK\x03\x04" + b"\x00" * 2000)

    _make_run(root / "run-a", "baseline", lr=0.001, val_acc=0.80)
    _make_run(root / "run-b", "lower-lr", lr=0.0003, val_acc=0.87)

    return {
        "csv": csv,
        "preds": preds,
        "script": script,
        "log": log,
        "model": model,
        "run_a": root / "run-a",
        "run_b": root / "run-b",
    }


def _call_every_read_only_tool(
    paths: dict[str, Path], *, project_path: Path
) -> list[dict[str, Any]]:
    """Run all six ledger-writing read-only tools against ``paths``."""
    return [
        mlcompass_advise(str(paths["csv"])),
        mlcompass_audit(str(paths["script"])),
        mlcompass_watch(str(paths["log"])),
        mlcompass_evaluate(str(paths["preds"])),
        mlcompass_deploy(str(paths["model"])),
        mlcompass_compare(
            str(paths["run_a"]),
            str(paths["run_b"]),
            project_path=str(project_path),
        ),
    ]


@pytest.fixture
def two_projects(tmp_path: Path) -> tuple[Path, Path]:
    """Two sibling mlcompass projects, A (active) and B (a bystander)."""
    a = tmp_path / "project-a"
    b = tmp_path / "project-b"
    a.mkdir()
    b.mkdir()
    ProjectContext.init("project-a", parent_dir=a)
    ProjectContext.init("project-b", parent_dir=b)
    return a, b


# --------------------------------------------------------------------------- #
# The reported defect                                                         #
# --------------------------------------------------------------------------- #


def test_read_only_tools_never_write_into_a_project_named_by_an_argument(
    two_projects: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """From project A, naming a file inside project B must not write into B.

    This is the reproduction from the README warning block: B's
    ``context.json`` grew from 152 to 583 bytes while A stayed untouched,
    with no ``..`` anywhere in the path.
    """
    a, b = two_projects
    monkeypatch.chdir(a)  # A is the active project.

    paths = _populate(b)
    before = _snapshot(b / ".mlcompass")

    _call_every_read_only_tool(paths, project_path=b)

    after = _snapshot(b / ".mlcompass")
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    assert changed == [], (
        f"tools declared read-only wrote into the bystander project {b}: {changed}"
    )


def test_ledger_write_lands_in_the_active_project(
    two_projects: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analysing a file that lives outside any project records it in A.

    The ledger is an audit trail of what *this* agent did, so work on a
    file that belongs to no project still belongs in the active one.
    """
    a, _b = two_projects
    monkeypatch.chdir(a)

    loose = a.parent / "loose"
    loose.mkdir()
    csv = loose / "data.csv"
    pd.DataFrame({"age": list(range(20)), "churn": [0, 1] * 10}).to_csv(csv, index=False)

    mlcompass_advise(str(csv))

    ctx = json.loads((a / ".mlcompass" / "context.json").read_text(encoding="utf-8"))
    assert [d["command"] for d in ctx["decisions"]] == ["advise"]
    assert ctx["target_column"] == "churn"
    log_lines = (a / ".mlcompass" / "advice.log").read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["command"] for line in log_lines] == ["advise"]


def test_cross_project_refusal_is_reported_to_the_caller(
    two_projects: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused write must be visible, not silently dropped.

    Otherwise the user runs ``advise``, sees a clean analysis, and only
    later notices ``status`` never recorded it.
    """
    a, b = two_projects
    monkeypatch.chdir(a)

    paths = _populate(b)
    result = mlcompass_advise(str(paths["csv"]))

    note = result.get("ledger")
    assert note is not None, "cross-project refusal was silent"
    assert note["written"] is False
    assert note["reason"] == "cross-project"
    assert str(b) in note["argument_project"]
    assert str(a) in (note["active_project"] or "")


def test_same_project_write_carries_no_refusal_note(
    two_projects: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The happy path keeps its existing result shape — no extra key."""
    a, _b = two_projects
    monkeypatch.chdir(a)

    paths = _populate(a)
    for result in _call_every_read_only_tool(paths, project_path=a):
        assert "ledger" not in result

    ctx = json.loads((a / ".mlcompass" / "context.json").read_text(encoding="utf-8"))
    commands = {d["command"] for d in ctx["decisions"]}
    assert commands == {"advise", "audit", "watch", "evaluate", "deploy", "compare"}


def test_no_active_project_still_refuses_an_argument_named_project(
    two_projects: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Running from outside any project does not licence writing into one.

    This is the ``mlcompass-mcp`` case: a client launches the server with
    the user's home directory as cwd, then asks about a file that happens
    to sit inside a project.
    """
    _a, b = two_projects
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    paths = _populate(b)
    before = _snapshot(b / ".mlcompass")

    result = mlcompass_advise(str(paths["csv"]))

    assert _snapshot(b / ".mlcompass") == before
    assert result["ledger"]["written"] is False
    assert result["ledger"]["active_project"] is None


# --------------------------------------------------------------------------- #
# The agent binds the boundary to its own --project-path                      #
# --------------------------------------------------------------------------- #


def test_run_agent_binds_the_ledger_root_to_its_project_path(
    two_projects: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``mlcompass agent --project-path A`` writes to A, whatever cwd is.

    The active project is the agent's own project, not the process's
    incidental working directory.
    """
    from mlcompass.agent import orchestrator
    from mlcompass.agent.backends import AgentResult
    from mlcompass.agent.tools import call_tool

    a, _b = two_projects
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    loose = tmp_path / "loose.csv"
    pd.DataFrame({"age": list(range(20)), "churn": [0, 1] * 10}).to_csv(loose, index=False)

    class _Backend:
        name = "fake"

        def run(self, task: str, **kwargs: Any) -> AgentResult:
            call_tool("mlcompass_advise", {"dataset_path": str(loose)})
            return AgentResult(ok=True, turns=1, final_text="done", stop_reason="end_turn")

    monkeypatch.setitem(orchestrator._BACKENDS, "fake", _Backend)

    orchestrator.run_agent(
        "analyse it",
        project_path=a,
        backend="fake",
        use_memory=False,
    )

    ctx = json.loads((a / ".mlcompass" / "context.json").read_text(encoding="utf-8"))
    assert [d["command"] for d in ctx["decisions"]] == ["advise"]


def test_ledger_root_is_restored_after_a_run(
    two_projects: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One agent run must not leave the boundary pointing at its project."""
    from mlcompass.agent import orchestrator
    from mlcompass.agent.backends import AgentResult

    a, b = two_projects
    monkeypatch.chdir(b)

    class _Backend:
        name = "fake"

        def run(self, task: str, **kwargs: Any) -> AgentResult:
            return AgentResult(ok=True, turns=1, final_text="done", stop_reason="end_turn")

    monkeypatch.setitem(orchestrator._BACKENDS, "fake", _Backend)
    orchestrator.run_agent("x", project_path=a, backend="fake", use_memory=False)

    # Back to cwd-derived: B is active again, so a write lands in B.
    paths = _populate(b)
    mlcompass_advise(str(paths["csv"]))
    ctx = json.loads((b / ".mlcompass" / "context.json").read_text(encoding="utf-8"))
    assert [d["command"] for d in ctx["decisions"]] == ["advise"]
