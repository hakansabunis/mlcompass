"""Tests for the MCP server surface.

The MCP server is an opt-in layer (extra ``mlcompass[mcp]``). If the
``mcp`` package is not installed we skip the whole module. Otherwise we
exercise every registered tool through its plain-Python callable — the
``@mcp.tool()`` decorator returns the function unchanged, so we can
invoke it without standing up a stdio transport.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

mcp_module = pytest.importorskip("mcp.server.fastmcp")

from mlcompass.mcp_server import (  # noqa: E402  (importorskip gate above)
    mcp,
    mlcompass_advise,
    mlcompass_audit,
    mlcompass_compare,
    mlcompass_deploy,
    mlcompass_evaluate,
    mlcompass_init,
    mlcompass_status,
    mlcompass_watch,
)

# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_eight_tools_registered() -> None:
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert names == {
        "mlcompass_init",
        "mlcompass_status",
        "mlcompass_advise",
        "mlcompass_audit",
        "mlcompass_watch",
        "mlcompass_compare",
        "mlcompass_evaluate",
        "mlcompass_deploy",
    }


@pytest.mark.asyncio
async def test_each_tool_has_input_schema_with_required_field() -> None:
    """Every tool exposes a JSON schema. Tools with required args must
    surface them; tools with all-default args ship a ``properties`` map.
    """
    tools = await mcp.list_tools()
    required_by_tool = {
        "mlcompass_init": {"name"},
        "mlcompass_advise": {"dataset_path"},
        "mlcompass_audit": {"script_path"},
        "mlcompass_watch": {"log_path"},
        "mlcompass_compare": {"run_a", "run_b"},
        "mlcompass_evaluate": {"results_path"},
        "mlcompass_deploy": {"model_path"},
        # status has only optional args
    }
    for tool in tools:
        schema = tool.inputSchema or {}
        assert schema.get("type") == "object"
        if tool.name in required_by_tool:
            required = set(schema.get("required") or [])
            assert required_by_tool[tool.name].issubset(required), (
                f"{tool.name} schema missing required args: "
                f"{required_by_tool[tool.name] - required}"
            )


# --------------------------------------------------------------------------- #
# mlcompass_init                                                              #
# --------------------------------------------------------------------------- #


def test_init_creates_project_directory(tmp_path: Path) -> None:
    result = mlcompass_init("demo", parent_dir=str(tmp_path))
    assert result["ok"] is True
    assert (tmp_path / ".mlcompass").is_dir()
    assert (tmp_path / ".mlcompass" / "project.yaml").is_file()
    assert (tmp_path / ".mlcompass" / "context.json").is_file()


def test_init_returns_error_envelope_on_duplicate(tmp_path: Path) -> None:
    mlcompass_init("demo", parent_dir=str(tmp_path))
    result = mlcompass_init("demo", parent_dir=str(tmp_path))
    assert result["ok"] is False
    assert result["error"] == "ProjectExistsError"
    assert "already exists" in result["message"].lower()


# --------------------------------------------------------------------------- #
# mlcompass_status                                                            #
# --------------------------------------------------------------------------- #


def test_status_fresh_project(tmp_path: Path) -> None:
    mlcompass_init("demo", parent_dir=str(tmp_path))
    result = mlcompass_status(project_path=str(tmp_path))
    assert result["ok"] is True
    assert result["project"]["name"] == "demo"
    assert result["state"]["project_type"] is None
    assert result["command_counts"] == {}
    assert result["decisions"] == []
    assert result["total_decisions"] == 0


def test_status_aggregates_advice_log(tmp_path: Path) -> None:
    mlcompass_init("demo", parent_dir=str(tmp_path))
    log_path = tmp_path / ".mlcompass" / "advice.log"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"timestamp": "2026-05-30T00:00:00Z", "command": "advise"}) + "\n")
        fh.write(json.dumps({"timestamp": "2026-05-30T00:00:01Z", "command": "advise"}) + "\n")
        fh.write(json.dumps({"timestamp": "2026-05-30T00:00:02Z", "command": "audit"}) + "\n")

    result = mlcompass_status(project_path=str(tmp_path))
    assert result["command_counts"] == {"advise": 2, "audit": 1}


def test_status_caps_recent_decisions(tmp_path: Path) -> None:
    mlcompass_init("demo", parent_dir=str(tmp_path))
    from mlcompass.context import ProjectContext

    project = ProjectContext.load(search_from=str(tmp_path))
    for i in range(8):
        project.append_decision(command="advise", summary=f"decision-{i}")

    result = mlcompass_status(project_path=str(tmp_path), recent_decisions=3)
    assert result["total_decisions"] == 8
    assert len(result["decisions"]) == 3
    # Last three should be the latest entries.
    assert [d["summary"] for d in result["decisions"]] == [
        "decision-5",
        "decision-6",
        "decision-7",
    ]


def test_status_returns_error_envelope_when_no_project(tmp_path: Path) -> None:
    result = mlcompass_status(project_path=str(tmp_path))
    assert result["ok"] is False
    assert result["error"] == "ProjectNotFoundError"


# --------------------------------------------------------------------------- #
# mlcompass_advise                                                            #
# --------------------------------------------------------------------------- #


def test_advise_returns_structured_analysis(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "age": [25, 31, 45, 22, 58],
            "income": [50_000.0, 80_000, 120_000, 35_000, 200_000],
            "churn": [0, 1, 0, 1, 0],
        }
    )
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)

    result = mlcompass_advise(str(csv))
    assert "shape" in result
    assert result["shape"]["rows"] == 5
    assert result["shape"]["cols"] == 3
    # Target detection should hit ``churn`` (high-confidence hint).
    assert result["target_hint"]["column"] == "churn"


def test_advise_error_envelope_for_missing_file(tmp_path: Path) -> None:
    result = mlcompass_advise(str(tmp_path / "nope.csv"))
    assert result["ok"] is False
    assert result["error"] in {"FileNotFoundError", "ValueError", "OSError"}


# --------------------------------------------------------------------------- #
# mlcompass_audit                                                             #
# --------------------------------------------------------------------------- #


_SCRIPT_WITH_ISSUES = """\
import torch
from torch.utils.data import DataLoader

model = torch.nn.LSTM(10, 20)
opt = torch.optim.Adam(model.parameters(), lr=0.001, momentum=0.9)
loader = DataLoader(dataset, batch_size=1)
"""


def test_audit_flags_known_rules(tmp_path: Path) -> None:
    src = tmp_path / "train.py"
    src.write_text(_SCRIPT_WITH_ISSUES, encoding="utf-8")

    result = mlcompass_audit(str(src))
    assert "findings" in result
    rule_ids = {f["rule_id"] for f in result["findings"]}
    # Multiple of the eight pure-AST rules should trigger.
    assert "seed" in rule_ids
    assert "optimizer" in rule_ids  # Adam with momentum=
    assert "dataloader" in rule_ids  # missing shuffle=


def test_audit_respects_skip_rules(tmp_path: Path) -> None:
    src = tmp_path / "train.py"
    src.write_text(_SCRIPT_WITH_ISSUES, encoding="utf-8")

    result = mlcompass_audit(str(src), skip_rules=["seed", "dataloader"])
    rule_ids = {f["rule_id"] for f in result["findings"]}
    assert "seed" not in rule_ids
    assert "dataloader" not in rule_ids


def test_audit_error_envelope_for_missing_file(tmp_path: Path) -> None:
    result = mlcompass_audit(str(tmp_path / "ghost.py"))
    assert result["ok"] is False


# --------------------------------------------------------------------------- #
# mlcompass_watch                                                             #
# --------------------------------------------------------------------------- #


def test_watch_plain_log_runs_detectors(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "Epoch 0: train_loss=0.80 val_loss=0.75",
                "Epoch 1: train_loss=0.50 val_loss=0.55",
                "Epoch 2: train_loss=0.35 val_loss=0.45",
                "Epoch 3: train_loss=0.20 val_loss=0.42",
                "Epoch 4: train_loss=0.10 val_loss=nan",
            ]
        ),
        encoding="utf-8",
    )

    result = mlcompass_watch(str(log))
    assert result["ok"] is True
    assert result["source"] == "plain_text"
    assert result["snapshot_count"] >= 5
    # NaN metric values should be normalised to None for JSON safety.
    assert all(
        v is None or isinstance(v, (int, float, type(None)))
        for snap in result["metrics"]
        for v in snap.values()
    )
    # The nan detector should still fire even with the value scrubbed.
    rule_ids = {f["rule_id"] for f in result["findings"]}
    assert "nan" in rule_ids


def test_watch_error_envelope_for_missing_file(tmp_path: Path) -> None:
    result = mlcompass_watch(str(tmp_path / "missing.log"))
    assert result["ok"] is False


# --------------------------------------------------------------------------- #
# mlcompass_compare                                                           #
# --------------------------------------------------------------------------- #


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


def test_compare_by_paths(tmp_path: Path) -> None:
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    _make_run(run_a, "baseline", lr=0.001, val_acc=0.80)
    _make_run(run_b, "lower-lr", lr=0.0003, val_acc=0.87)

    result = mlcompass_compare(str(run_a), str(run_b))
    assert "verdict" in result or "metric_comparison" in result
    # Differing lr should appear in the config diff.
    diff_keys = {entry.get("key") for entry in result.get("config_diff", [])}
    assert "lr" in diff_keys


def test_compare_error_envelope_for_missing_run(tmp_path: Path) -> None:
    result = mlcompass_compare(
        str(tmp_path / "nope-a"),
        str(tmp_path / "nope-b"),
    )
    assert result["ok"] is False
    assert result["error"] == "RunNotFoundError"


# --------------------------------------------------------------------------- #
# mlcompass_evaluate                                                          #
# --------------------------------------------------------------------------- #


def test_evaluate_binary_classification(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_prob = np.where(
        y_true == 1,
        rng.uniform(0.55, 0.95, size=n),
        rng.uniform(0.05, 0.45, size=n),
    )
    y_pred = (y_prob >= 0.5).astype(int)
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob})
    csv = tmp_path / "preds.csv"
    df.to_csv(csv, index=False)

    result = mlcompass_evaluate(str(csv))
    assert result["task"] == "binary_classification"
    assert "metrics" in result
    assert "confusion_matrix" in result
    assert "threshold_sweep" in result
    assert result["metrics"]["auc"] > 0.5


def test_evaluate_leakage_smell_warning_fires(tmp_path: Path) -> None:
    """Perfect predictions on 200 rows should trigger the leakage-smell."""
    n = 200
    y_true = np.array([0, 1] * (n // 2))
    y_prob = y_true.astype(float) * 0.99 + 0.005  # near-perfect probs
    y_pred = (y_prob >= 0.5).astype(int)
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob})
    csv = tmp_path / "preds.csv"
    df.to_csv(csv, index=False)

    result = mlcompass_evaluate(str(csv))
    joined = " ".join(result.get("warnings", [])).lower()
    assert "leakage" in joined or "too good" in joined or "implausibly" in joined


def test_evaluate_error_envelope_for_missing_file(tmp_path: Path) -> None:
    result = mlcompass_evaluate(str(tmp_path / "nope.csv"))
    assert result["ok"] is False


# --------------------------------------------------------------------------- #
# mlcompass_deploy                                                            #
# --------------------------------------------------------------------------- #


def test_deploy_inspects_model_file(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    # PyTorch .pt files start with the PK\x03\x04 ZIP magic.
    model.write_bytes(b"PK\x03\x04" + b"\x00" * 2000)

    result = mlcompass_deploy(str(model))
    assert "model" in result
    assert "checklist" in result
    assert "warnings" in result


def test_deploy_error_envelope_for_missing_model(tmp_path: Path) -> None:
    result = mlcompass_deploy(str(tmp_path / "ghost.pt"))
    assert result["ok"] is False


def test_deploy_rejects_unknown_target(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"PK\x03\x04" + b"\x00" * 100)

    result = mlcompass_deploy(str(model), target="azure-functions")
    assert result["ok"] is False
    assert "azure-functions" in result["message"].lower()
