"""Anthropic-API backend — the default driver.

Hand-rolled tool-use loop on top of ``anthropic.Anthropic.messages.create``.
This is the universal backend: it only needs ``ANTHROPIC_API_KEY``, with
no other binaries or daemons. It costs only the model's output tokens
and has no rate-limit surface beyond Anthropic's own quotas.

The loop:

1. Send the user task + system prompt + tool registry.
2. Read the model's response. For every text / thinking block, fire an
   ``AgentStep`` so the UI streams.
3. For every ``tool_use`` block, ask the permission callback (if the
   tool is marked ``mutates``), dispatch via :func:`tools.call_tool`,
   and append the result back into the conversation.
4. Repeat until the model emits ``stop_reason == "end_turn"``, the loop
   hits ``max_turns``, or a fatal error fires.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from .._system_prompt import SYSTEM_PROMPT
from ..tools import BY_NAME, anthropic_tool_specs, call_tool
from ._common import AgentResult, AgentStep, PermissionCallback, StepCallback

_LOG = logging.getLogger(__name__)


class AnthropicAPIBackend:
    """Reference backend talking directly to the Anthropic Messages API."""

    name = "api"

    def __init__(self, *, max_tokens: int = 4096) -> None:
        self.max_tokens = max_tokens

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
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The anthropic-api backend requires the 'anthropic' package. "
                "Install with: pip install 'mlcompass[agent]'"
            ) from e

        client = anthropic.Anthropic()
        system = system_prompt or SYSTEM_PROMPT
        tool_specs = anthropic_tool_specs()

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": _build_initial_prompt(task=task, project_path=project_path),
            },
        ]

        for turn in range(1, max_turns + 1):
            try:
                # The anthropic SDK ships strict TypedDicts for the tool
                # and message schemas; our hand-built dicts validate at
                # runtime but mypy can't reconcile them, so we cross the
                # SDK boundary as Any.
                response = client.messages.create(
                    model=model,
                    max_tokens=self.max_tokens,
                    system=system,
                    tools=cast(Any, tool_specs),
                    messages=cast(Any, messages),
                )
            except Exception as e:  # noqa: BLE001 — surface any API error uniformly.
                _LOG.exception("Anthropic API error on turn %d", turn)
                on_step(AgentStep(kind="stop", content=str(e)))
                return AgentResult(
                    ok=False,
                    turns=turn,
                    final_text="",
                    stop_reason="api_error",
                    error=str(e),
                )

            # Stream every content block and harvest tool_use blocks for
            # the next round of dispatching. We duck-type via getattr +
            # the runtime "type" tag so this code stays compatible across
            # anthropic-SDK versions whose union of block types changes
            # — mypy can't narrow the SDK's content union through that
            # string check, so we keep the locals as Any.
            tool_uses: list[Any] = []
            assistant_text_parts: list[str] = []
            for raw_block in response.content:
                block: Any = raw_block
                btype = getattr(block, "type", None)
                if btype == "text":
                    text = getattr(block, "text", "")
                    assistant_text_parts.append(text)
                    on_step(AgentStep(kind="message", content=text))
                elif btype == "thinking":
                    on_step(AgentStep(kind="thinking", content=getattr(block, "thinking", "")))
                elif btype == "tool_use":
                    tool_uses.append(block)
                    on_step(
                        AgentStep(
                            kind="tool_call",
                            tool_name=block.name,
                            tool_input=dict(block.input),
                        )
                    )

            # No tool calls + end_turn ⇒ the agent is done.
            if response.stop_reason in {"end_turn", "stop_sequence"} and not tool_uses:
                final_text = "\n".join(assistant_text_parts).strip()
                on_step(AgentStep(kind="stop", content=final_text))
                return AgentResult(
                    ok=True,
                    turns=turn,
                    final_text=final_text,
                    stop_reason=response.stop_reason,
                )

            # Persist the assistant turn (whether it had tools or not) so
            # the next call can see history.
            messages.append({"role": "assistant", "content": response.content})

            # Dispatch every tool_use block and feed the results back.
            if tool_uses:
                tool_results: list[dict[str, Any]] = []
                for use in tool_uses:
                    result = _dispatch_with_permission(
                        use=use,
                        on_permission=on_permission,
                        on_step=on_step,
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": use.id,
                            "content": json.dumps(result, default=str),
                        }
                    )
                messages.append({"role": "user", "content": tool_results})

        # Loop exited via max-turns.
        on_step(AgentStep(kind="stop", content="max_turns reached"))
        return AgentResult(
            ok=False,
            turns=max_turns,
            final_text="",
            stop_reason="max_turns",
            error=f"Agent did not converge within {max_turns} turns.",
        )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _dispatch_with_permission(
    use: Any,
    *,
    on_permission: PermissionCallback,
    on_step: StepCallback,
) -> dict[str, Any]:
    """Run a single tool_use block, honouring the mutation permission gate."""
    name = use.name
    arguments = dict(use.input)
    spec = BY_NAME.get(name)
    if spec is not None and spec.mutates and not on_permission(name, arguments):
        denied = {
            "ok": False,
            "error": "PermissionDenied",
            "message": (f"User declined to run {name}. Adjust the plan or ask differently."),
        }
        on_step(
            AgentStep(
                kind="tool_result",
                tool_name=name,
                tool_input=arguments,
                tool_result=denied,
            )
        )
        return denied

    result = call_tool(name, arguments)
    on_step(
        AgentStep(
            kind="tool_result",
            tool_name=name,
            tool_input=arguments,
            tool_result=result,
        )
    )
    return result


def _build_initial_prompt(*, task: str, project_path: str) -> str:
    return (
        f"<task>\n{task.strip()}\n</task>\n\n"
        f"<project_path>{project_path}</project_path>\n\n"
        "Reason carefully about what to do, then call the right "
        "mlcompass_* tool. When you have enough information to give the "
        "user a useful answer, write the final summary in your own words "
        "and stop calling tools."
    )
