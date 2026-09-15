"""``--provider`` / ``--base-url`` on every command that has ``--llm``.

Eight commands take ``--llm``: advise, audit, watch, compare, evaluate,
deploy, monitor, optimize. Each of them must be able to point at an
OpenAI-compatible endpoint, and each must keep working unchanged when the
user passes nothing.

The gate that used to read "no ANTHROPIC_API_KEY, skip the LLM step" is the
subtle part: against a local ollama server there is no key to set and never
will be, so a key check that is not provider-aware would skip the very
configuration this change exists to enable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from mlcompass import cli as cli_module
from mlcompass.cli import cli

LOCAL_BASE_URL = "http://localhost:11434/v1"
LOCAL_MODEL = "qwen2.5:7b"


# --------------------------------------------------------------------------- #
# Fixtures — one minimal input per command                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _no_inherited_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "MLCOMPASS_LLM_PROVIDER",
        "MLCOMPASS_LLM_BASE_URL",
        "MLCOMPASS_LLM_MODEL",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def command_inputs(tmp_path: Path) -> dict[str, list[str]]:
    """Build a minimal invocation for each ``--llm``-capable command."""
    # advise
    dataset = tmp_path / "data.csv"
    pd.DataFrame(
        {"age": list(range(40)), "churn": [0, 1] * 20},
    ).to_csv(dataset, index=False)

    # audit
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n"
        "import torch.nn as nn\n"
        "model = nn.LSTM(10, 20)\n"
        "opt = torch.optim.Adam(model.parameters(), lr=1e-3, momentum=0.9)\n",
        encoding="utf-8",
    )

    # watch
    log = tmp_path / "train.log"
    lines = [f"Epoch {i} train_loss={0.8 - i * 0.1:.4f}" for i in range(6)]
    lines.append("Epoch 6 train_loss=nan")
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # compare + optimize
    def _make_run(parent: Path, name: str, cfg: dict[str, Any], metrics: list[dict]) -> Path:
        run = parent / name
        run.mkdir(parents=True, exist_ok=True)
        (run / "config.yaml").write_text(
            yaml.safe_dump({"name": name, "config": cfg}), encoding="utf-8"
        )
        (run / "metrics.json").write_text(json.dumps({"metrics": metrics}), encoding="utf-8")
        return run

    runs_dir = tmp_path / "runs"
    run_a = _make_run(runs_dir, "r1", {"lr": 0.01}, [{"epoch": 0, "val_acc": 0.60}])
    run_b = _make_run(runs_dir, "r2", {"lr": 0.001}, [{"epoch": 0, "val_acc": 0.75}])
    _make_run(runs_dir, "r3", {"lr": 0.0001}, [{"epoch": 0, "val_acc": 0.85}])

    # evaluate
    rng = np.random.default_rng(0)
    n = 80
    y_true = rng.integers(0, 2, size=n)
    y_prob = np.where(
        y_true == 1,
        rng.uniform(0.55, 0.95, size=n),
        rng.uniform(0.05, 0.45, size=n),
    )
    preds = tmp_path / "preds.csv"
    pd.DataFrame(
        {"y_true": y_true, "y_pred": (y_prob >= 0.5).astype(int), "y_prob": y_prob}
    ).to_csv(preds, index=False)

    # deploy
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"PK\x03\x04" + b"\x00" * 200)

    # monitor — mild shift, so the command still exits 0
    mrng = np.random.default_rng(7)
    ref = tmp_path / "ref.csv"
    cur = tmp_path / "cur.csv"
    pd.DataFrame({"x": mrng.normal(0, 1, 500)}).to_csv(ref, index=False)
    pd.DataFrame({"x": mrng.normal(0, 1, 500)}).to_csv(cur, index=False)

    return {
        "advise": ["advise", str(dataset)],
        "audit": ["audit", str(script)],
        "watch": ["watch", str(log)],
        "compare": ["compare", str(run_a), str(run_b)],
        "evaluate": ["evaluate", str(preds)],
        "deploy": ["deploy", str(model_file)],
        "monitor": ["monitor", str(ref), str(cur)],
        "optimize": ["optimize", "--runs-dir", str(runs_dir), "--metric", "val_acc"],
    }


#: command name -> (cli module attribute holding the agent callable, reply)
COMMAND_AGENTS: dict[str, tuple[str, dict[str, Any]]] = {
    "advise": (
        "_advisor_callable",
        {"models": [{"name": "XGBoost", "reason": "tabular"}], "features": [], "pitfalls": []},
    ),
    "audit": (
        "_audit_prioritizer_callable",
        {
            "priorities": [{"rule_id": "seed", "priority_rank": 1, "blast_radius": "x"}],
            "synthesis": "Fix the seed first.",
        },
    ),
    "watch": (
        "_watch_diagnostician_callable",
        {
            "diagnosis": [
                {
                    "finding_rule_id": "nan",
                    "hypothesis": "lr too high",
                    "recommended_action": "lower lr",
                    "confidence": "high",
                }
            ],
            "summary": "Diverged at epoch 6.",
        },
    ),
    "compare": (
        "_compare_hypothesizer_callable",
        {"hypothesis": "lower lr won", "key_factors": [], "next_experiment": "try 3e-4"},
    ),
    "evaluate": (
        "_evaluate_interpreter_callable",
        {
            "assessment": "Strong ranking.",
            "strengths": ["AUC high"],
            "weaknesses": ["precision"],
            "next_steps": ["sweep threshold"],
        },
    ),
    "deploy": (
        "_deploy_advisor_callable",
        {
            "verdict": "Ready for canary deploy.",
            "blockers": [],
            "next_steps": ["Add monitoring."],
            "rollout_strategy": "Canary 5%.",
        },
    ),
    "monitor": (
        "_monitor_interpreter_callable",
        {"headline": "Stable.", "likely_cause": "nothing changed", "next_steps": ["a", "b"]},
    ),
    "optimize": (
        "_optimize_strategist_callable",
        {"headline": "lr dominates", "pattern": "monotone", "next_plan": ["drop lr", "hold bs"]},
    ),
}

COMMANDS = sorted(COMMAND_AGENTS)


def _install_recorder(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> dict[str, Any]:
    """Replace the command's agent callable with a kwargs recorder."""
    attr, reply = COMMAND_AGENTS[command]
    captured: dict[str, Any] = {"called": False, "kwargs": {}}

    def fake(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        captured["called"] = True
        captured["kwargs"] = kwargs
        return reply

    monkeypatch.setattr(cli_module, attr, fake)
    # evaluate also narrates leakage when the smell fires; stub it so no
    # test in this file can reach a real provider.
    monkeypatch.setattr(
        cli_module,
        "_leakage_investigator_callable",
        lambda evidence, **_kw: {
            "verdict": "leakage_uncertain",
            "confidence": "low",
            "evidence_cited": [],
            "primary_hypothesis": "(stub)",
            "recommended_checks": [],
        },
    )
    return captured


# --------------------------------------------------------------------------- #
# The flags exist on every --llm command                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("command", COMMANDS)
def test_command_exposes_provider_and_base_url(command: str) -> None:
    result = CliRunner().invoke(cli, [command, "--help"])
    assert result.exit_code == 0, result.output
    assert "--provider" in result.output
    assert "--base-url" in result.output
    # Requirement 3: --model keeps working.
    assert "--model" in result.output


# --------------------------------------------------------------------------- #
# A local, keyless endpoint is a complete configuration                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("command", COMMANDS)
def test_llm_runs_against_a_local_endpoint_without_any_key(
    command: str,
    command_inputs: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No ANTHROPIC_API_KEY, no OPENAI_API_KEY — ollama still works."""
    captured = _install_recorder(monkeypatch, command)

    args = [
        *command_inputs[command],
        "--llm",
        "--provider",
        "openai",
        "--base-url",
        LOCAL_BASE_URL,
        "--model",
        LOCAL_MODEL,
    ]
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, args)

    assert result.exit_code == 0, result.output
    assert captured["called"], result.output
    assert captured["kwargs"]["provider"] == "openai"
    assert captured["kwargs"]["base_url"] == LOCAL_BASE_URL
    assert captured["kwargs"]["model"] == LOCAL_MODEL


@pytest.mark.parametrize("command", COMMANDS)
def test_default_invocation_asks_for_no_particular_provider(
    command: str,
    command_inputs: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Plain ``--llm`` forwards no overrides, so the agent default applies."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    captured = _install_recorder(monkeypatch, command)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, [*command_inputs[command], "--llm"])

    assert result.exit_code == 0, result.output
    assert captured["called"], result.output
    assert captured["kwargs"]["provider"] is None
    assert captured["kwargs"]["base_url"] is None
    assert captured["kwargs"]["model"] is None


@pytest.mark.parametrize("command", COMMANDS)
def test_explicit_model_still_reaches_the_agent(
    command: str,
    command_inputs: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    captured = _install_recorder(monkeypatch, command)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli, [*command_inputs[command], "--llm", "--model", "claude-haiku-4-5"]
        )

    assert result.exit_code == 0, result.output
    assert captured["kwargs"]["model"] == "claude-haiku-4-5"


# --------------------------------------------------------------------------- #
# The gate stays honest                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("command", COMMANDS)
def test_openai_without_key_or_base_url_is_skipped_and_named(
    command: str,
    command_inputs: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``--provider openai`` with nothing to reach: skip, and say which key."""
    captured = _install_recorder(monkeypatch, command)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, [*command_inputs[command], "--llm", "--provider", "openai"])

    assert result.exit_code == 0, result.output
    assert not captured["called"]
    assert "openai_api_key" in result.output.lower()


@pytest.mark.parametrize("command", COMMANDS)
def test_anthropic_without_key_still_warns_about_anthropic(
    command: str,
    command_inputs: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The pre-existing message is unchanged for the default provider."""
    captured = _install_recorder(monkeypatch, command)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, [*command_inputs[command], "--llm"])

    assert result.exit_code == 0, result.output
    assert not captured["called"]
    assert "anthropic_api_key" in result.output.lower()


@pytest.mark.parametrize("command", COMMANDS)
def test_environment_provider_is_honoured_without_flags(
    command: str,
    command_inputs: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``MLCOMPASS_LLM_*`` alone is enough; the CLI gate must not block it."""
    monkeypatch.setenv("MLCOMPASS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MLCOMPASS_LLM_BASE_URL", LOCAL_BASE_URL)
    monkeypatch.setenv("MLCOMPASS_LLM_MODEL", LOCAL_MODEL)
    captured = _install_recorder(monkeypatch, command)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, [*command_inputs[command], "--llm"])

    assert result.exit_code == 0, result.output
    assert captured["called"], result.output


def test_provider_choice_rejects_an_unknown_value(
    command_inputs: dict[str, list[str]],
) -> None:
    result = CliRunner().invoke(cli, [*command_inputs["advise"], "--llm", "--provider", "bedrock"])
    assert result.exit_code != 0
    assert "bedrock" in result.output


def test_evaluate_forwards_provider_to_the_leakage_investigator(
    command_inputs: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``evaluate --llm`` narrates leakage too; it must follow the same target.

    ``agents/leakage_investigator.py`` is the published artifact and is not
    modified — the CLI hands it a client built for the selected provider.
    """
    _install_recorder(monkeypatch, "evaluate")
    seen: dict[str, Any] = {}

    def fake_leakage(evidence: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {
            "verdict": "leakage_uncertain",
            "confidence": "low",
            "evidence_cited": [],
            "primary_hypothesis": "(stub)",
            "recommended_checks": [],
        }

    monkeypatch.setattr(cli_module, "_leakage_investigator_callable", fake_leakage)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            [
                *command_inputs["evaluate"],
                "--llm",
                "--provider",
                "openai",
                "--base-url",
                LOCAL_BASE_URL,
                "--model",
                LOCAL_MODEL,
            ],
        )

    assert result.exit_code == 0, result.output
    assert seen.get("provider") == "openai"
    assert seen.get("base_url") == LOCAL_BASE_URL
