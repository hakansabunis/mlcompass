"""Claude Code backend — opt-in driver via ``claude-agent-sdk``.

Routes the agent loop through the user's locally installed Claude Code
CLI. The SDK takes care of:

- multi-turn conversation persistence,
- tool-call streaming + permission flow,
- token budgets and rate-limit handling.

What we contribute:

- one ``SdkMcpTool`` per mlcompass tool, sharing the registry with the
  Anthropic-API backend so a bug fix lands in both backends at once,
- a ``can_use_tool`` callback that delegates the mutation gate to our
  ``PermissionCallback``,
- translation of SDK message types into our ``AgentStep`` events.

This backend requires the ``claude-agent-sdk`` package AND the
``claude`` CLI on PATH. If either is missing the import raises a clear
error message pointing at the install instructions.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .._system_prompt import SYSTEM_PROMPT
from ..tools import BY_NAME, TOOL_REGISTRY, call_tool
from ._common import AgentResult, AgentStep, PermissionCallback, StepCallback

_LOG = logging.getLogger(__name__)


class ClaudeCodeBackend:
    """Backend talking to Anthropic via the local Claude Code CLI."""

    name = "claude-code"

    # ------------------------------------------------------------------ #
    # Public surface                                                     #
    # ------------------------------------------------------------------ #

    def run(
        self,
        task: str,
        *,
        project_path: str,
        max_turns: int,
        model: str,
        on_permission: PermissionCallback,
        on_step: StepCallback,
        system_prompt: str | None = None,
    ) -> AgentResult:
        try:
            import claude_agent_sdk as sdk
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The claude-code backend requires the 'claude-agent-sdk' "
                "package AND the 'claude' CLI on PATH. "
                "Install with: pip install 'mlcompass[agent-claude-code]'"
            ) from e

        sdk_tools = _build_sdk_tools(sdk)
        mcp_server = sdk.create_sdk_mcp_server(
            name="mlcompass",
            version="0.5.0",
            tools=sdk_tools,
        )

        # The SDK's permission callback uses its own typed result classes,
        # so we wrap our user-facing PermissionCallback to translate.
        async def can_use_tool(
            tool_name: str,
            tool_input: dict[str, Any],
            _context: Any,
        ) -> Any:
            spec = BY_NAME.get(_strip_namespace(tool_name))
            # Read-only tools auto-allow; the SDK still calls us so we
            # can record the event, but we never block them.
            if spec is None or not spec.mutates:
                return sdk.PermissionResultAllow()
            allowed = on_permission(_strip_namespace(tool_name), tool_input)
            if allowed:
                return sdk.PermissionResultAllow()
            return sdk.PermissionResultDeny(
                message=f"User declined to run {tool_name}.",
                interrupt=False,
            )

        options = sdk.ClaudeAgentOptions(
            mcp_servers={"mlcompass": mcp_server},
            allowed_tools=[f"mcp__mlcompass__{spec.name}" for spec in TOOL_REGISTRY],
            system_prompt=system_prompt or SYSTEM_PROMPT,
            model=model,
            max_turns=max_turns,
            permission_mode="default",
            can_use_tool=can_use_tool,
            cwd=project_path,
        )

        return asyncio.run(_run_async(task=task, options=options, on_step=on_step, sdk=sdk))


# --------------------------------------------------------------------------- #
# Async loop                                                                  #
# --------------------------------------------------------------------------- #


async def _run_async(
    *,
    task: str,
    options: Any,
    on_step: StepCallback,
    sdk: Any,
) -> AgentResult:
    """Iterate over the SDK's message stream and translate to AgentStep."""
    final_text_parts: list[str] = []
    turns = 0
    stop_reason = "end_turn"
    error: str | None = None
    ok = True

    try:
        async for message in sdk.query(prompt=task, options=options):
            mtype = type(message).__name__
            if mtype == "AssistantMessage":
                turns += 1
                for block in getattr(message, "content", []) or []:
                    btype = type(block).__name__
                    if btype == "TextBlock":
                        text = getattr(block, "text", "")
                        final_text_parts.append(text)
                        on_step(AgentStep(kind="message", content=text))
                    elif btype == "ThinkingBlock":
                        on_step(
                            AgentStep(
                                kind="thinking",
                                content=getattr(block, "thinking", ""),
                            )
                        )
                    elif btype == "ToolUseBlock":
                        on_step(
                            AgentStep(
                                kind="tool_call",
                                tool_name=_strip_namespace(getattr(block, "name", "")),
                                tool_input=dict(getattr(block, "input", {}) or {}),
                            )
                        )
            elif mtype == "UserMessage":
                # Tool results come back from the SDK as user messages
                # with ToolResultBlock content blocks.
                for block in getattr(message, "content", []) or []:
                    if type(block).__name__ == "ToolResultBlock":
                        result = _extract_tool_result(block)
                        on_step(
                            AgentStep(
                                kind="tool_result",
                                tool_result=result,
                            )
                        )
            elif mtype == "ResultMessage":
                stop_reason = str(getattr(message, "subtype", "") or "end_turn")
    except Exception as e:  # noqa: BLE001
        _LOG.exception("claude-code backend error")
        error = str(e)
        ok = False
        stop_reason = "sdk_error"

    final_text = "\n".join(p for p in final_text_parts if p.strip()).strip()
    on_step(AgentStep(kind="stop", content=final_text or (error or "")))
    return AgentResult(
        ok=ok,
        turns=turns,
        final_text=final_text,
        stop_reason=stop_reason,
        error=error,
    )


# --------------------------------------------------------------------------- #
# Tool wrapping                                                               #
# --------------------------------------------------------------------------- #


def _build_sdk_tools(sdk: Any) -> list[Any]:
    """Wrap every mlcompass tool as an ``SdkMcpTool``.

    The SDK's ``@tool`` decorator wraps an async callable that takes a
    dict and returns a dict. Our dispatchers are sync — we delegate
    through :func:`tools.call_tool` to keep one execution path for both
    backends.
    """
    sdk_tools = []
    for spec in TOOL_REGISTRY:
        # Bind ``spec_name`` at definition time so the closure captures
        # the right value instead of the loop variable.
        wrapped = _make_async_wrapper(spec.name)
        decorated = sdk.tool(
            name=spec.name,
            description=spec.description,
            input_schema=spec.input_schema,
        )(wrapped)
        sdk_tools.append(decorated)
    return sdk_tools


def _make_async_wrapper(spec_name: str) -> Any:
    """Return an async callable that dispatches by ``spec_name``."""

    async def _wrapper(arguments: dict[str, Any]) -> dict[str, Any]:
        # The SDK expects the standard ``content`` envelope used by
        # FastMCP / official MCP servers; we translate our flat dict
        # into that shape so the model reads results consistently.
        result = call_tool(spec_name, arguments)
        import json

        return {
            "content": [{"type": "text", "text": json.dumps(result, default=str)}],
        }

    _wrapper.__name__ = f"_dispatch_{spec_name}"
    return _wrapper


def _strip_namespace(tool_name: str) -> str:
    """Convert ``mcp__mlcompass__mlcompass_advise`` → ``mlcompass_advise``.

    The SDK prefixes tools with ``mcp__<server>__`` when they live in
    an MCP server. Our permission callback and downstream UI want the
    bare mlcompass name.
    """
    if tool_name.startswith("mcp__mlcompass__"):
        return tool_name[len("mcp__mlcompass__") :]
    return tool_name


def _extract_tool_result(block: Any) -> dict[str, Any]:
    """Pull the dict payload out of an SDK ``ToolResultBlock``."""
    content = getattr(block, "content", None)
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text") or ""
                try:
                    import json

                    parsed: dict[str, Any] = json.loads(text)
                except (ValueError, TypeError):
                    return {"ok": True, "raw": text}
                return parsed
    if isinstance(content, str):
        return {"ok": True, "raw": content}
    return {"ok": True, "raw": ""}
