"""Claude-Code backend tests with a fully mocked ``claude_agent_sdk``.

We never spawn the real Claude Code CLI. Instead we install a fake
SDK module exposing the names the backend imports, plus an async
``query`` generator scripted to emit a sequence of typed messages.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

# Imported at collection time, before any test swaps a fake module into
# ``sys.modules["claude_agent_sdk"]``. ``_real_sdk_types`` is the genuine
# SDK; we use it to render the argv the CLI would actually be exec'd with.
import claude_agent_sdk as _real_sdk_types
import pytest
from claude_agent_sdk._internal.transport.subprocess_cli import (
    SubprocessCLITransport as _RealSubprocessCLITransport,
)

from mlcompass.agent.backends import AgentStep
from mlcompass.agent.tools import TOOL_REGISTRY

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


@pytest.fixture(autouse=True)
def _restore_real_sdk() -> Any:
    """Undo ``install_fake_sdk``'s ``sys.modules`` swap after every test.

    Without this the fake module leaks into every later test in the
    session, so any test that needs the genuine SDK (e.g. to render the
    CLI argv) silently gets the stub instead.
    """
    saved = sys.modules.get("claude_agent_sdk")
    yield
    if saved is not None:
        sys.modules["claude_agent_sdk"] = saved
    else:  # pragma: no cover — SDK is a dev dependency, always importable
        sys.modules.pop("claude_agent_sdk", None)


def _render_cli_argv(allowed_tools: list[str]) -> list[str]:
    """Render the argv the SDK would exec for these ``allowed_tools``.

    Builds a genuine ``ClaudeAgentOptions`` and asks the genuine
    ``SubprocessCLITransport`` to build its command line. ``cli_path`` is
    stubbed so no CLI discovery (or subprocess) happens.
    """
    options = _real_sdk_types.ClaudeAgentOptions(
        allowed_tools=list(allowed_tools),
        system_prompt="stub",
        model="claude-sonnet-4-5",
        max_turns=5,
        permission_mode="default",
        cli_path="/stub/claude",
    )
    transport = _RealSubprocessCLITransport(prompt="stub", options=options)
    return transport._build_command()


def _allowed_tools_from_argv(argv: list[str]) -> list[str]:
    """Pull the comma-joined ``--allowedTools`` value out of rendered argv."""
    if "--allowedTools" not in argv:
        return []
    value = argv[argv.index("--allowedTools") + 1]
    return [part for part in value.split(",") if part]


def _run_backend_capturing_options(
    *,
    on_permission: Any = lambda *_: True,
) -> Any:
    """Run the backend against a trivial fake stream; return the fake SDK."""
    fake = install_fake_sdk([_ResultMessage(subtype="end_turn")])

    from mlcompass.agent.backends.claude_code import ClaudeCodeBackend

    ClaudeCodeBackend().run(
        task="x",
        project_path=".",
        max_turns=5,
        model="claude-sonnet-4-5",
        on_permission=on_permission,
        on_step=lambda _: None,
    )
    return fake


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_mutating_tool_absent_from_rendered_allowed_tools_argv() -> None:
    """The SDK auto-runs anything in ``--allowedTools`` without prompting.

    Its own docs for ``allowed_tools``: "Tool names that are auto-allowed
    without prompting for permission. These tools execute automatically
    without asking the user for approval." So listing ``mlcompass_init``
    there means the permission callback is never consulted for the one
    tool it exists to gate. Assert against the argv the SDK would really
    exec, not against our own option object.
    """
    fake = _run_backend_capturing_options()
    allowed = list(fake._last["options"].allowed_tools)

    argv = _render_cli_argv(allowed)
    rendered = _allowed_tools_from_argv(argv)

    assert "mcp__mlcompass__mlcompass_init" not in rendered, (
        "mlcompass_init is auto-approved by the SDK; the permission prompt "
        f"can never fire. Rendered --allowedTools: {rendered}"
    )
    # Every read-only tool must still be auto-allowed — the gate is for
    # mutation, not for slowing down reads.
    for spec in TOOL_REGISTRY:
        if not spec.mutates:
            assert f"mcp__mlcompass__{spec.name}" in rendered


def test_permission_callback_is_reached_for_the_mutating_tool() -> None:
    """Drive the gate the way the SDK does: skip anything auto-allowed.

    The SDK does not invoke ``can_use_tool`` for a tool listed in
    ``allowed_tools``. This test reproduces that rule, then checks the
    mutating tool still reaches our callback and that a refusal denies.
    """
    asked: list[tuple[str, dict[str, Any]]] = []

    def _deny(name: str, args: dict[str, Any]) -> bool:
        asked.append((name, args))
        return False

    fake = _run_backend_capturing_options(on_permission=_deny)
    opts = fake._last["options"]
    allowed = set(opts.allowed_tools)

    import asyncio

    async def _gate(tool_name: str, tool_input: dict[str, Any]) -> Any:
        # The SDK's rule: auto-allowed tools never reach can_use_tool.
        if tool_name in allowed:
            return "auto-allowed-by-sdk"
        return await opts.can_use_tool(tool_name, tool_input, None)

    outcome = asyncio.run(
        _gate("mcp__mlcompass__mlcompass_init", {"name": "churn", "parent_dir": "."})
    )

    assert asked == [("mlcompass_init", {"name": "churn", "parent_dir": "."})], (
        "the permission callback was never consulted for the mutating tool; "
        f"it was auto-allowed by the SDK instead (outcome={outcome!r})"
    )
    assert getattr(outcome, "behavior", None) == "deny"

    # A read-only tool goes the other way: auto-allowed, callback untouched.
    read_only = asyncio.run(_gate("mcp__mlcompass__mlcompass_status", {}))
    assert read_only == "auto-allowed-by-sdk"
    assert len(asked) == 1


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
    # Seven auto-allowed tools, namespaced under mcp__mlcompass__ — the
    # eighth (mlcompass_init) is deliberately withheld so the SDK routes
    # it through can_use_tool instead of running it unprompted.
    assert all(name.startswith("mcp__mlcompass__") for name in opts.allowed_tools)
    expected = {f"mcp__mlcompass__{spec.name}" for spec in TOOL_REGISTRY if not spec.mutates}
    assert set(opts.allowed_tools) == expected
    assert len(opts.allowed_tools) == 7


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
