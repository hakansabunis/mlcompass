"""Anthropic-API backend tests with a fully mocked ``anthropic.Anthropic`` client.

We never hit the real API in tests. Instead we install a fake client
that returns a scripted sequence of ``messages.create`` responses,
mimicking the model emitting text + tool_use blocks and eventually
stopping with ``stop_reason == "end_turn"``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mlcompass.agent.backends import AgentStep
from mlcompass.agent.backends.anthropic_api import AnthropicAPIBackend

# --------------------------------------------------------------------------- #
# Fake Anthropic client                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class _FakeResponse:
    content: list[Any]
    stop_reason: str


class _FakeMessages:
    def __init__(self, scripted: Iterable[_FakeResponse]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if not self._scripted:
            raise RuntimeError("Fake client ran out of scripted responses")
        return self._scripted.pop(0)


class _FakeAnthropic:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.messages = _FakeMessages([])


@pytest.fixture
def fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> _FakeAnthropic:
    """Replace ``import anthropic`` with a module exposing the fake client."""
    fake_mod = type(sys)("anthropic")

    instance_holder: dict[str, _FakeAnthropic] = {}

    class _Anthropic(_FakeAnthropic):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            instance_holder["last"] = self

    fake_mod.Anthropic = _Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)
    return instance_holder  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Behaviour                                                                   #
# --------------------------------------------------------------------------- #


def _run(
    backend: AnthropicAPIBackend,
    *,
    task: str,
    project_path: str,
    scripted: list[_FakeResponse],
    fake_holder: Any,
    on_permission: Any = lambda name, args: True,
    on_step: Any = None,
) -> Any:
    """Invoke ``backend.run`` with a pre-loaded scripted reply queue."""
    events: list[AgentStep] = []

    def _collect(step: AgentStep) -> None:
        events.append(step)
        if on_step is not None:
            on_step(step)

    result = backend.run(
        task=task,
        project_path=project_path,
        max_turns=10,
        model="claude-sonnet-4-5",
        on_permission=on_permission,
        on_step=_collect,
    )
    # Stash for assertions on the model-facing payload.
    last = fake_holder["last"]
    last.messages._scripted = []  # tidy
    return result, events, last.messages.calls


def test_text_only_response_terminates_after_one_turn(
    fake_anthropic: Any,
) -> None:
    """If the model emits a single text block and stops, we exit cleanly."""
    backend = AnthropicAPIBackend()

    # Inject scripted responses by patching messages.create on the
    # instance that will be constructed inside backend.run.
    def _seed(*_: Any, **__: Any) -> _FakeAnthropic:
        client = _FakeAnthropic()
        client.messages = _FakeMessages(
            [
                _FakeResponse(
                    content=[_FakeTextBlock(text="Try XGBoost first.")],
                    stop_reason="end_turn",
                )
            ]
        )
        fake_anthropic["last"] = client
        return client

    import sys as _sys

    _sys.modules["anthropic"].Anthropic = _seed  # type: ignore[attr-defined]

    result, events, calls = _run(
        backend,
        task="Look at this data",
        project_path=".",
        scripted=[],
        fake_holder=fake_anthropic,
    )

    assert result.ok is True
    assert result.turns == 1
    assert "XGBoost" in result.final_text
    kinds = [e.kind for e in events]
    assert "message" in kinds
    assert kinds[-1] == "stop"
    assert len(calls) == 1


def test_tool_call_is_dispatched_and_result_round_trips(
    tmp_path: Path,
    fake_anthropic: Any,
) -> None:
    """Model emits a tool_use, we run it, the result is fed back."""
    csv = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2, 3], "churn": [0, 1, 0]}).to_csv(csv, index=False)

    backend = AnthropicAPIBackend()

    scripted = [
        _FakeResponse(
            content=[
                _FakeToolUseBlock(
                    id="t-1",
                    name="mlcompass_advise",
                    input={"dataset_path": str(csv)},
                )
            ],
            stop_reason="tool_use",
        ),
        _FakeResponse(
            content=[_FakeTextBlock(text="The dataset looks like binary classification.")],
            stop_reason="end_turn",
        ),
    ]

    def _seed(*_: Any, **__: Any) -> _FakeAnthropic:
        client = _FakeAnthropic()
        client.messages = _FakeMessages(scripted)
        fake_anthropic["last"] = client
        return client

    import sys as _sys

    _sys.modules["anthropic"].Anthropic = _seed  # type: ignore[attr-defined]

    result, events, calls = _run(
        backend,
        task="Advise on this",
        project_path=str(tmp_path),
        scripted=[],
        fake_holder=fake_anthropic,
    )

    assert result.ok is True
    assert result.turns == 2
    assert "binary classification" in result.final_text

    tool_calls = [e for e in events if e.kind == "tool_call"]
    tool_results = [e for e in events if e.kind == "tool_result"]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "mlcompass_advise"
    assert len(tool_results) == 1
    assert tool_results[0].tool_result is not None
    assert tool_results[0].tool_result["target_hint"]["column"] == "churn"

    # Second messages.create call should carry both the assistant
    # turn AND the tool_result back to the model.
    second_call = calls[1]
    assert second_call["messages"][-1]["role"] == "user"


def test_permission_denial_short_circuits_mutating_tool(
    tmp_path: Path,
    fake_anthropic: Any,
) -> None:
    """A user 'no' on mlcompass_init returns a denial envelope to the model."""
    backend = AnthropicAPIBackend()

    scripted = [
        _FakeResponse(
            content=[
                _FakeToolUseBlock(
                    id="t-1",
                    name="mlcompass_init",
                    input={"name": "demo", "parent_dir": str(tmp_path)},
                )
            ],
            stop_reason="tool_use",
        ),
        _FakeResponse(
            content=[_FakeTextBlock(text="OK, skipping init.")],
            stop_reason="end_turn",
        ),
    ]

    def _seed(*_: Any, **__: Any) -> _FakeAnthropic:
        client = _FakeAnthropic()
        client.messages = _FakeMessages(scripted)
        fake_anthropic["last"] = client
        return client

    import sys as _sys

    _sys.modules["anthropic"].Anthropic = _seed  # type: ignore[attr-defined]

    permission_calls: list[tuple[str, dict[str, Any]]] = []

    def deny(name: str, args: dict[str, Any]) -> bool:
        permission_calls.append((name, args))
        return False

    result, events, _ = _run(
        backend,
        task="Init a new project",
        project_path=str(tmp_path),
        scripted=[],
        fake_holder=fake_anthropic,
        on_permission=deny,
    )

    assert result.ok is True  # graceful conversation completion
    assert permission_calls == [("mlcompass_init", {"name": "demo", "parent_dir": str(tmp_path)})]
    # The init was blocked: no .mlcompass/ created.
    assert not (tmp_path / ".mlcompass").exists()
    tool_results = [e for e in events if e.kind == "tool_result"]
    assert tool_results[0].tool_result is not None
    assert tool_results[0].tool_result["error"] == "PermissionDenied"


def test_max_turns_returns_failed_result(fake_anthropic: Any) -> None:
    """A model that never says end_turn must hit the safety cap."""
    backend = AnthropicAPIBackend()

    # Each scripted response has a tool_use ⇒ loop continues.
    looped = [
        _FakeResponse(
            content=[
                _FakeToolUseBlock(
                    id=f"t-{i}",
                    name="mlcompass_status",
                    input={"project_path": "."},
                )
            ],
            stop_reason="tool_use",
        )
        for i in range(20)
    ]

    def _seed(*_: Any, **__: Any) -> _FakeAnthropic:
        client = _FakeAnthropic()
        client.messages = _FakeMessages(looped)
        fake_anthropic["last"] = client
        return client

    import sys as _sys

    _sys.modules["anthropic"].Anthropic = _seed  # type: ignore[attr-defined]

    events: list[AgentStep] = []

    result = backend.run(
        task="Loop forever",
        project_path=".",
        max_turns=3,
        model="claude-sonnet-4-5",
        on_permission=lambda *_: True,
        on_step=events.append,
    )

    assert result.ok is False
    assert result.stop_reason == "max_turns"
    assert result.turns == 3
    assert events[-1].kind == "stop"


def test_api_error_surfaces_clean_failure(fake_anthropic: Any) -> None:
    backend = AnthropicAPIBackend()

    class _BoomMessages:
        def create(self, **_: Any) -> Any:
            raise RuntimeError("Anthropic upstream 503")

    def _seed(*_: Any, **__: Any) -> _FakeAnthropic:
        client = _FakeAnthropic()
        client.messages = _BoomMessages()  # type: ignore[assignment]
        fake_anthropic["last"] = client
        return client

    import sys as _sys

    _sys.modules["anthropic"].Anthropic = _seed  # type: ignore[attr-defined]

    events: list[AgentStep] = []
    result = backend.run(
        task="kaboom",
        project_path=".",
        max_turns=5,
        model="claude-sonnet-4-5",
        on_permission=lambda *_: True,
        on_step=events.append,
    )

    assert result.ok is False
    assert result.stop_reason == "api_error"
    assert "503" in (result.error or "")
