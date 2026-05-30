"""Tests for the per-run transcript writer."""

from __future__ import annotations

import json
from pathlib import Path

from mlcompass.agent.backends import AgentResult, AgentStep
from mlcompass.agent.transcript import (
    TranscriptWriter,
    new_run_dir,
    write_summary,
)


def test_new_run_dir_creates_unique_subdir(tmp_path: Path) -> None:
    first = new_run_dir(tmp_path)
    second = new_run_dir(tmp_path)
    assert first.parent.name == "agent_runs"
    assert first.is_dir() and second.is_dir()
    assert first != second


def test_transcript_writer_appends_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "agent_runs" / "demo" / "transcript.jsonl"
    with TranscriptWriter(path) as writer:
        writer.write_step(AgentStep(kind="message", content="hello"))
        writer.write_step(
            AgentStep(
                kind="tool_call",
                tool_name="mlcompass_advise",
                tool_input={"dataset_path": "data.csv"},
            )
        )
        writer.write_step(
            AgentStep(
                kind="tool_result",
                tool_name="mlcompass_advise",
                tool_result={"ok": True, "shape": {"rows": 5}},
            )
        )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    kinds = [entry["kind"] for entry in parsed]
    assert kinds == ["message", "tool_call", "tool_result"]
    assert "timestamp" in parsed[0]


def test_transcript_truncates_giant_tool_results(tmp_path: Path) -> None:
    """A multi-megabyte tool_result must be capped before hitting disk."""
    path = tmp_path / "transcript.jsonl"
    giant_payload = {"ok": True, "blob": "x" * 50_000}
    with TranscriptWriter(path) as writer:
        writer.write_step(
            AgentStep(
                kind="tool_result",
                tool_name="mlcompass_advise",
                tool_result=giant_payload,
            )
        )

    entry = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert entry["tool_result"]["_truncated"] is True
    assert entry["tool_result"]["_original_size"] > 50_000


def test_write_summary_records_metadata(tmp_path: Path) -> None:
    run_dir = new_run_dir(tmp_path)
    result = AgentResult(
        ok=True,
        turns=3,
        final_text="Recommended XGBoost as a baseline.",
        stop_reason="end_turn",
    )
    summary_path = write_summary(
        run_dir,
        task="What model should I try?",
        result=result,
        backend_name="api",
        model="claude-sonnet-4-5",
    )

    text = summary_path.read_text(encoding="utf-8")
    assert "What model should I try?" in text
    assert "Recommended XGBoost" in text
    assert "claude-sonnet-4-5" in text
    assert "end_turn" in text
    assert "Turns**: 3" in text
