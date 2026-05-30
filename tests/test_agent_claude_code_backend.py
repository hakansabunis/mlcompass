"""Claude-Code backend tests with a fully mocked ``claude_agent_sdk``.

We never spawn the real Claude Code CLI. Instead we install a fake
SDK module exposing the names the backend imports, plus an async
``query`` generator scripted to emit a sequence of typed messages.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import pytest

from mlcompass.agent.backends import AgentStep

# --------------------------------------------------------------------------- #
# Fake claude_agent_sdk                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class _TextBlock:
    text: str
    __class_name__: str = "TextBlock"

    def __post_init__(self) -> None:
        type(self).__name__ = "TextBlock"


@dataclass
class _ToolUseBlock:
    name: str
    input: dict[str, Any]
    __class_name__: str = "ToolUseBlock"

    def __post_init__(self) -> None:
        type(self).__name__ = "ToolUseBlock"


@dataclass
class _ToolResultBlock:
    content: list[Any]
    __class_name__: str = "ToolResultBlock"

    def __post_init__(self) -> None:
        type(self).__name__ = "ToolResultBlock"


@dataclass
class _AssistantMessage:
    content: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        type(self).__name__ = "AssistantMessage"


@dataclass
class _UserMessage:
    content: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        type(self).__name__ = "UserMessage"


@dataclass
class _ResultMessage:
    subtype: str = "end_turn"

    def __post_init__(self) -> None:
        type(self).__name__ = "ResultMessage"


@dataclass
class _PermissionResultAllow:
    behavior: str = "allow"


@dataclass
class _PermissionResultDeny:
    message: str = ""
    interrupt: bool = False
    behavior: str = "deny"


@dataclass
class _ClaudeAgentOptions:
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    system_prompt: str = ""
    model: str = ""
    max_turns: int = 0
    permission_mode: str = "default"
    can_use_tool: Any = None
    cwd: str = "."


def _tool_factory(name: str, description: str, input_schema: dict[str, Any]) -> Any:
    """Stand-in for ``sdk.tool(...)`` decorator factory."""

    def _decorator(fn: Any) -> Any:
        fn.__sdk_tool_name__ = name
        fn.__sdk_tool_desc__ = description
        fn.__sdk_tool_schema__ = input_schema
        return fn

    return _decorator


def _create_sdk_mcp_server(*, name: str, version: str, tools: list[Any]) -> dict[str, Any]:
    return {"name": name, "version": version, "tools": tools}


def install_fake_sdk(scripted_messages: list[Any]) -> Any:
    """Install a fake ``claude_agent_sdk`` module backed by ``scripted_messages``.

    Returns the fake module so tests can introspect, e.g., which
    options were passed.
    """
    fake = type(sys)("claude_agent_sdk")
    fake.TextBlock = _TextBlock  # type: ignore[attr-defined]
    fake.ToolUseBlock = _ToolUseBlock  # type: ignore[attr-defined]
    fake.ToolResultBlock = _ToolResultBlock  # type: ignore[attr-defined]
    fake.AssistantMessage = _AssistantMessage  # type: ignore[attr-defined]
    fake.UserMessage = _UserMessage  # type: ignore[attr-defined]
    fake.ResultMessage = _ResultMessage  # type: ignore[attr-defined]
    fake.PermissionResultAllow = _PermissionResultAllow  # type: ignore[attr-defined]
    fake.PermissionResultDeny = _PermissionResultDeny  # type: ignore[attr-defined]
    fake.ClaudeAgentOptions = _ClaudeAgentOptions  # type: ignore[attr-defined]
    fake.tool = _tool_factory  # type: ignore[attr-defined]
    fake.create_sdk_mcp_server = _create_sdk_mcp_server  # type: ignore[attr-defined]

    last_options_holder: dict[str, Any] = {}

    async def _query(*, prompt: str, options: Any, transport: Any = None) -> Any:
        last_options_holder["options"] = options
        last_options_holder["prompt"] = prompt
        for message in scripted_messages:
            yield message

    fake.query = _query  # type: ignore[attr-defined]
    fake._last = last_options_holder  # type: ignore[attr-defined]

    sys.modules["claude_agent_sdk"] = fake
    return fake


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_text_only_response_terminates(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install_fake_sdk(
        [
            _AssistantMessage(content=[_TextBlock(text="Try XGBoost first.")]),
            _ResultMessage(subtype="end_turn"),
        ]
    )

    from mlcompass.agent.backends.claude_code import ClaudeCodeBackend

    backend = ClaudeCodeBackend()
    events: list[AgentStep] = []

    result = backend.run(
        task="Look at this data",
        project_path=".",
        max_turns=5,
        model="claude-sonnet-4-5",
        on_permission=lambda *_: True,
        on_step=events.append,
    )

    assert result.ok is True
    assert "XGBoost" in result.final_text
    assert result.stop_reason == "end_turn"
    kinds = [e.kind for e in events]
    assert "message" in kinds
    assert kinds[-1] == "stop"

    opts = fake._last["options"]
    # Eight allowed tools, namespaced under mcp__mlcompass__.
    assert all(name.startswith("mcp__mlcompass__") for name in opts.allowed_tools)
    assert len(opts.allowed_tools) == 8


def test_tool_use_and_result_flow_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm tool_use messages become AgentStep tool_call events."""
    scripted = [
        _AssistantMessage(
            content=[
                _ToolUseBlock(
                    name="mcp__mlcompass__mlcompass_advise",
                    input={"dataset_path": "data.csv"},
                )
            ]
        ),
        _UserMessage(
            content=[
                _ToolResultBlock(
                    content=[
                        {
                            "type": "text",
                            "text": '{"ok": true, "shape": {"rows": 100}}',
                        }
                    ]
                )
            ]
        ),
        _AssistantMessage(content=[_TextBlock(text="Looks balanced.")]),
        _ResultMessage(subtype="end_turn"),
    ]
    install_fake_sdk(scripted)

    from mlcompass.agent.backends.claude_code import ClaudeCodeBackend

    backend = ClaudeCodeBackend()
    events: list[AgentStep] = []

    result = backend.run(
        task="Advise",
        project_path=".",
        max_turns=5,
        model="claude-sonnet-4-5",
        on_permission=lambda *_: True,
        on_step=events.append,
    )

    assert result.ok is True
    assert "balanced" in result.final_text
    tool_calls = [e for e in events if e.kind == "tool_call"]
    tool_results = [e for e in events if e.kind == "tool_result"]
    assert len(tool_calls) == 1
    # Namespace stripped for the UI.
    assert tool_calls[0].tool_name == "mlcompass_advise"
    assert tool_results[0].tool_result == {"ok": True, "shape": {"rows": 100}}


def test_missing_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If claude_agent_sdk is missing the backend tells the user how to install."""
    # Pretend the SDK is uninstalled.
    monkeypatch.delitem(sys.modules, "claude_agent_sdk", raising=False)
    import builtins

    real_import = builtins.__import__

    def _block(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "claude_agent_sdk":
            raise ImportError("No module named 'claude_agent_sdk'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block)

    from mlcompass.agent.backends.claude_code import ClaudeCodeBackend

    backend = ClaudeCodeBackend()
    with pytest.raises(ImportError, match=r"mlcompass\[agent-claude-code\]"):
        backend.run(
            task="x",
            project_path=".",
            max_turns=1,
            model="claude-sonnet-4-5",
            on_permission=lambda *_: True,
            on_step=lambda _: None,
        )


def test_sdk_error_surfaces_clean_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exceptions during the async loop become a typed failure result."""
    fake = install_fake_sdk([])  # empty stream

    async def _boom(*, prompt: str, options: Any, transport: Any = None) -> Any:
        raise RuntimeError("subprocess died")
        # Make the function recognised as async-gen by yielding once.
        yield  # pragma: no cover

    fake.query = _boom  # type: ignore[attr-defined]

    from mlcompass.agent.backends.claude_code import ClaudeCodeBackend

    backend = ClaudeCodeBackend()
    events: list[AgentStep] = []

    result = backend.run(
        task="x",
        project_path=".",
        max_turns=1,
        model="claude-sonnet-4-5",
        on_permission=lambda *_: True,
        on_step=events.append,
    )

    assert result.ok is False
    assert result.stop_reason == "sdk_error"
    assert "subprocess died" in (result.error or "")
