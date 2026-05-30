"""Tests for the agent memory layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mlcompass.agent.memory import (
    MemoryBlock,
    find_run_dir,
    list_recent_runs,
    load_decisions_summary,
    load_recent_context,
    load_run_summary,
    load_transcript,
    transcript_to_text,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _seed_context(project_path: Path, decisions: list[dict[str, Any]]) -> None:
    (project_path / "context.json").write_text(
        json.dumps({"decisions": decisions}, indent=2),
        encoding="utf-8",
    )


def _seed_transcript(run_dir: Path, lines: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    transcript = run_dir / "transcript.jsonl"
    with transcript.open("w", encoding="utf-8") as fh:
        for entry in lines:
            fh.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------- #
# MemoryBlock.as_prompt_prefix                                                #
# --------------------------------------------------------------------------- #


def test_memory_block_renders_all_three_sections() -> None:
    block = MemoryBlock(
        headline="Cross-session memory.",
        decisions_summary="- [2026-05-30] advise: try XGBoost",
        prior_run_summary="Agent ran advise + audit on churn data.",
        source_run_id="20260530-110000",
    )
    text = block.as_prompt_prefix()
    assert "<memory_headline>" in text
    assert "<recent_decisions>" in text
    assert "id='20260530-110000'" in text
    assert "<prior_agent_run" in text


def test_memory_block_empty_renders_empty_string() -> None:
    block = MemoryBlock(
        headline="",
        decisions_summary="",
        prior_run_summary="",
    )
    assert block.as_prompt_prefix() == ""


# --------------------------------------------------------------------------- #
# load_decisions_summary                                                      #
# --------------------------------------------------------------------------- #


def test_load_decisions_summary_returns_compact_lines(tmp_path: Path) -> None:
    _seed_context(
        tmp_path,
        [
            {"timestamp": "2026-05-30T10:00:00Z", "command": "advise", "summary": "Try XGBoost"},
            {"timestamp": "2026-05-31T10:00:00Z", "command": "audit", "summary": "2 errors"},
        ],
    )
    text = load_decisions_summary(tmp_path)
    assert "- [2026-05-30] advise: Try XGBoost" in text
    assert "- [2026-05-31] audit: 2 errors" in text


def test_load_decisions_summary_caps_with_limit(tmp_path: Path) -> None:
    _seed_context(
        tmp_path,
        [
            {"timestamp": "2026-05-30T10:00:00Z", "command": "advise", "summary": f"d{i}"}
            for i in range(20)
        ],
    )
    text = load_decisions_summary(tmp_path, limit=3)
    assert text.count("\n") == 2  # 3 lines, 2 newlines
    # The TAIL (latest) should be kept — d19, d18, d17.
    assert "d19" in text
    assert "d17" in text
    assert "d0" not in text


def test_load_decisions_summary_missing_context_returns_empty(tmp_path: Path) -> None:
    assert load_decisions_summary(tmp_path) == ""


def test_load_decisions_summary_bad_json_returns_empty(tmp_path: Path) -> None:
    (tmp_path / "context.json").write_text("not json", encoding="utf-8")
    assert load_decisions_summary(tmp_path) == ""


# --------------------------------------------------------------------------- #
# Transcript I/O                                                              #
# --------------------------------------------------------------------------- #


def test_load_transcript_parses_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "message", "content": "hi"}) + "\n")
        fh.write(json.dumps({"kind": "stop", "content": "bye"}) + "\n")
    steps = load_transcript(path)
    assert [s["kind"] for s in steps] == ["message", "stop"]


def test_load_transcript_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_transcript(tmp_path / "nope.jsonl") == []


def test_load_transcript_tolerates_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "message", "content": "hi"}) + "\n")
        fh.write("garbage non-json\n")
        fh.write(json.dumps({"kind": "stop", "content": "ok"}) + "\n")
    assert len(load_transcript(path)) == 2


def test_transcript_to_text_renders_each_kind() -> None:
    steps = [
        {"kind": "message", "content": "hello"},
        {"kind": "tool_call", "tool_name": "mlcompass_advise", "tool_input": {"x": 1}},
        {"kind": "tool_result", "tool_result": {"ok": True, "shape": {"rows": 5}}},
        {"kind": "stop", "content": "final"},
    ]
    text = transcript_to_text(steps)
    assert "agent: hello" in text
    assert "→ mlcompass_advise" in text
    assert "← ok" in text
    assert "stop: final" in text


def test_transcript_to_text_truncates_at_max_chars() -> None:
    """Oversized payloads must be capped before the summariser sees them."""
    big = [{"kind": "message", "content": "x" * 5000} for _ in range(10)]
    text = transcript_to_text(big, max_chars=200)
    assert "truncated" in text


# --------------------------------------------------------------------------- #
# Run-dir discovery                                                           #
# --------------------------------------------------------------------------- #


def test_find_run_dir_by_full_id(tmp_path: Path) -> None:
    target = tmp_path / "agent_runs" / "20260530-120000"
    target.mkdir(parents=True)
    assert find_run_dir(tmp_path, "20260530-120000") == target


def test_find_run_dir_by_prefix(tmp_path: Path) -> None:
    target = tmp_path / "agent_runs" / "20260530-120000-02"
    target.mkdir(parents=True)
    assert find_run_dir(tmp_path, "20260530-120000") == target


def test_find_run_dir_missing_returns_none(tmp_path: Path) -> None:
    assert find_run_dir(tmp_path, "anything") is None


def test_list_recent_runs_returns_newest_first(tmp_path: Path) -> None:
    base = tmp_path / "agent_runs"
    for name in ("20260528-100000", "20260530-090000", "20260529-110000"):
        (base / name).mkdir(parents=True)
    runs = list_recent_runs(tmp_path, limit=2)
    assert [r.name for r in runs] == ["20260530-090000", "20260529-110000"]


# --------------------------------------------------------------------------- #
# High-level memory builders                                                  #
# --------------------------------------------------------------------------- #


def test_load_run_summary_skips_summariser_when_disabled(tmp_path: Path) -> None:
    run_dir = tmp_path / "agent_runs" / "20260530-150000"
    _seed_transcript(
        run_dir,
        [
            {"kind": "message", "content": "hello"},
            {"kind": "stop", "content": "bye"},
        ],
    )
    _seed_context(
        tmp_path,
        [{"timestamp": "2026-05-30T10:00:00Z", "command": "advise", "summary": "Try XGB"}],
    )

    block = load_run_summary(
        tmp_path,
        "20260530-150000",
        summarise=False,
    )
    # No summariser ⇒ prior_run_summary IS the raw transcript text.
    assert "agent: hello" in block.prior_run_summary
    assert block.source_run_id == "20260530-150000"
    assert "advise" in block.decisions_summary


def test_load_run_summary_missing_run_returns_friendly_block(tmp_path: Path) -> None:
    block = load_run_summary(tmp_path, "nonexistent", summarise=False)
    assert "could not find" in block.headline.lower()
    assert block.prior_run_summary == ""


def test_load_recent_context_pulls_decisions_and_last_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "agent_runs" / "20260530-150000"
    _seed_transcript(run_dir, [{"kind": "stop", "content": "done"}])
    _seed_context(
        tmp_path,
        [{"timestamp": "2026-05-30T09:00:00Z", "command": "advise", "summary": "XGB"}],
    )

    block = load_recent_context(tmp_path, summarise=False)
    assert "Cross-session memory" in block.headline
    assert "XGB" in block.decisions_summary
    assert "done" in block.prior_run_summary
    assert block.source_run_id == "20260530-150000"


def test_load_recent_context_handles_empty_project(tmp_path: Path) -> None:
    """No decisions, no prior runs ⇒ empty block, no crash."""
    block = load_recent_context(tmp_path, summarise=False)
    assert block.headline == ""
    assert block.decisions_summary == ""
    assert block.prior_run_summary == ""


def test_summariser_returns_friendly_string_on_no_agentlite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If agentlite is absent we must NOT crash the orchestrator."""
    import builtins
    import sys as _sys

    monkeypatch.delitem(_sys.modules, "agentlite", raising=False)
    real_import = builtins.__import__

    def _block(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "agentlite":
            raise ImportError("no agentlite")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block)

    from mlcompass.agent.memory import summarise_transcript_with_agentlite

    text = summarise_transcript_with_agentlite("agent: hi")
    assert "agentlite" in text.lower()
