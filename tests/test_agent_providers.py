"""Provider independence for the eight pure-JSON reasoner agents.

``leakage_investigator`` already spoke both the Anthropic Messages API and
the OpenAI Chat Completions API. The other eight agents went through
``agentlite.Agent``, which only knows ``client.messages.create`` — so
``advise --llm`` and friends could only ever reach Anthropic.

These tests pin three things:

1. **Anthropic stays the default and stays byte-identical.** With no
   arguments and no environment, every agent must send exactly the request
   agentlite sent before the change (model, ``max_tokens=4096``, cached
   system block, one user message, no ``tools`` key).
2. **Each agent can reach an OpenAI-compatible endpoint**, with chat
   completions and a JSON response — no tool binding, because these agents
   make no tool calls.
3. **Resolution order** is explicit argument, then environment, then the
   agent's own built-in default.

Everything here is deterministic: the Anthropic path uses
``agentlite.testing.MockClient`` and the OpenAI path uses the local
``FakeOpenAIClient`` below. No test in this file makes a network call.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest

from mlcompass.agents import _common
from mlcompass.agents.advise import ADVISOR_MODEL_DEFAULT, get_recommendation
from mlcompass.agents.audit import prioritize_findings
from mlcompass.agents.compare import hypothesize_comparison
from mlcompass.agents.deploy import advise_deployment
from mlcompass.agents.evaluate import interpret_evaluation
from mlcompass.agents.monitor import interpret_drift
from mlcompass.agents.optimize import strategize_optimize
from mlcompass.agents.watch import diagnose_findings

# --------------------------------------------------------------------------- #
# Fake OpenAI-compatible client                                               #
# --------------------------------------------------------------------------- #


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls: list[Any] = []


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = "stop"


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, owner: FakeOpenAIClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> _FakeCompletion:
        self._owner.calls.append(kwargs)
        if self._owner.reject_once is not None:
            message = self._owner.reject_once
            self._owner.reject_once = None
            raise RuntimeError(message)
        return _FakeCompletion(self._owner.content)


class _FakeChat:
    def __init__(self, owner: FakeOpenAIClient) -> None:
        self.completions = _FakeCompletions(owner)


class FakeOpenAIClient:
    """Minimal stand-in for ``openai.OpenAI`` (chat completions only).

    ``reject_once`` makes the first call raise with the given message, so a
    test can exercise the "endpoint does not support this parameter"
    fallback without a real server.
    """

    def __init__(self, content: str, *, reject_once: str | None = None) -> None:
        self.content = content
        self.reject_once = reject_once
        self.calls: list[dict[str, Any]] = []
        self.chat = _FakeChat(self)

    @property
    def last_call(self) -> dict[str, Any]:
        return self.calls[-1]


# --------------------------------------------------------------------------- #
# The eight agents, with a minimal valid input and a valid response           #
# --------------------------------------------------------------------------- #

SAMPLE_ANALYSIS: dict[str, Any] = {
    "path": "data.csv",
    "shape": {"rows": 100, "cols": 3},
    "columns": [{"name": "churn", "type": "categorical", "cardinality": 2}],
    "target_hint": {"column": "churn", "confidence": "high"},
    "task_hint": {"type": "binary_classification"},
    "warnings": [],
}

SAMPLE_AUDIT: dict[str, Any] = {
    "path": "train.py",
    "findings": [{"rule_id": "seed", "severity": "error", "message": "No seed.", "line": None}],
}

SAMPLE_SNAPSHOTS: list[dict[str, Any]] = [
    {"epoch": 0, "step": None, "metrics": {"train_loss": 0.5, "val_loss": 0.5}},
    {"epoch": 5, "step": None, "metrics": {"train_loss": 0.1, "val_loss": 0.55}},
]

SAMPLE_WATCH_FINDINGS: list[dict[str, Any]] = [
    {"rule_id": "overfitting", "severity": "warning", "message": "train down, val up."}
]

SAMPLE_COMPARISON: dict[str, Any] = {
    "run_a": {"id": "a"},
    "run_b": {"id": "b"},
    "config_diff": [{"key": "lr", "a": 0.01, "b": 0.001}],
    "metric_comparison": [],
    "verdict": "b_better",
}

SAMPLE_EVALUATION: dict[str, Any] = {
    "task": "binary_classification",
    "metrics": {"auc": 0.94, "f1": 0.81},
    "warnings": [],
}

SAMPLE_DEPLOY: dict[str, Any] = {
    "model": {"format": "joblib", "size_mb": 3.1},
    "target": {"name": "lambda"},
    "checklist": [],
    "warnings": [],
}

SAMPLE_DRIFT: dict[str, Any] = {
    "reference_rows": 1000,
    "current_rows": 900,
    "features": ["age"],
    "aggregate": {"mean_psi": 0.31, "max_psi": 0.31},
    "top_drifted": ["age"],
    "feature_results": [{"feature": "age", "kind": "numeric", "psi": 0.31, "severity": "major"}],
    "verdict": {"status": "major_drift"},
}

SAMPLE_OPTIMIZE: dict[str, Any] = {
    "metric": "val_acc",
    "direction": "max",
    "n_runs": 4,
    "n_scored": 4,
    "best": {"run_id": "r3", "score": 0.91},
    "leaderboard": [{"run_id": "r3", "score": 0.91}],
    "sensitivity": [{"param": "lr", "spearman": -0.8}],
    "suggestions": [{"lr": 0.003}],
}


def _resp(payload: dict[str, Any]) -> str:
    return json.dumps(payload)


# (id, callable, positional args, valid JSON response)
AGENT_CASES: list[tuple[str, Any, tuple[Any, ...], str]] = [
    (
        "advise",
        get_recommendation,
        (SAMPLE_ANALYSIS,),
        _resp(
            {
                "models": [{"name": "XGBoost", "reason": "tabular"}],
                "features": [],
                "pitfalls": [],
            }
        ),
    ),
    (
        "audit",
        prioritize_findings,
        (SAMPLE_AUDIT,),
        _resp(
            {
                "priorities": [{"rule_id": "seed", "priority_rank": 1, "blast_radius": "x"}],
                "synthesis": "Fix the seed.",
            }
        ),
    ),
    (
        "watch",
        diagnose_findings,
        (SAMPLE_SNAPSHOTS, SAMPLE_WATCH_FINDINGS),
        _resp(
            {
                "diagnosis": [
                    {
                        "finding_rule_id": "overfitting",
                        "hypothesis": "too much capacity",
                        "recommended_action": "raise dropout",
                        "confidence": "high",
                    }
                ],
                "summary": "Overfitting from epoch 4.",
            }
        ),
    ),
    (
        "compare",
        hypothesize_comparison,
        (SAMPLE_COMPARISON,),
        _resp({"hypothesis": "lower lr won", "key_factors": [], "next_experiment": "try 0.0005"}),
    ),
    (
        "evaluate",
        interpret_evaluation,
        (SAMPLE_EVALUATION,),
        _resp(
            {
                "assessment": "Strong ranking.",
                "strengths": ["AUC 0.94"],
                "weaknesses": ["F1 0.81"],
                "next_steps": ["sweep the threshold"],
            }
        ),
    ),
    (
        "deploy",
        advise_deployment,
        (SAMPLE_DEPLOY,),
        _resp(
            {
                "verdict": "Nearly ready.",
                "blockers": [],
                "next_steps": ["pin deps"],
                "rollout_strategy": "canary",
            }
        ),
    ),
    (
        "monitor",
        interpret_drift,
        (SAMPLE_DRIFT,),
        _resp(
            {
                "headline": "age drifted hard",
                "likely_cause": "upstream pipeline change",
                "next_steps": ["check the ETL", "retrain"],
            }
        ),
    ),
    (
        "optimize",
        strategize_optimize,
        (SAMPLE_OPTIMIZE,),
        _resp(
            {
                "headline": "lr dominates",
                "pattern": "monotone in lr",
                "next_plan": ["drop lr to 0.003", "hold batch size"],
            }
        ),
    ),
]

AGENT_IDS = [case[0] for case in AGENT_CASES]


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No inherited MLCOMPASS_/provider environment leaks into these tests."""
    for var in (
        "MLCOMPASS_LLM_PROVIDER",
        "MLCOMPASS_LLM_BASE_URL",
        "MLCOMPASS_LLM_MODEL",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #
# 1. Anthropic stays the default, byte-identical                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("name", "fn", "args", "response"), AGENT_CASES, ids=AGENT_IDS)
def test_default_path_sends_the_unchanged_anthropic_request(
    name: str,
    fn: Any,
    args: tuple[Any, ...],
    response: str,
) -> None:
    """An upgrading user who changes nothing gets the identical request.

    The expected shape is what ``agentlite.Agent.run`` sent before provider
    support existed: exactly four keys, 4096 max tokens, the system prompt
    as a single cached text block, one user message, and no ``tools`` key
    (these agents are pure reasoners).
    """
    from agentlite.testing import MockClient, text_response

    client = MockClient(responses=[text_response(response)])
    fn(*args, client=client)

    kwargs = client.last_create_kwargs
    assert sorted(kwargs) == ["max_tokens", "messages", "model", "system"]
    assert kwargs["max_tokens"] == 4096
    assert kwargs["model"].startswith("claude-")
    assert isinstance(kwargs["system"], list)
    assert kwargs["system"][0]["type"] == "text"
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert [m["role"] for m in kwargs["messages"]] == ["user"]
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


@pytest.mark.parametrize(("name", "fn", "args", "response"), AGENT_CASES, ids=AGENT_IDS)
def test_explicit_anthropic_provider_matches_the_default(
    name: str,
    fn: Any,
    args: tuple[Any, ...],
    response: str,
) -> None:
    """``provider="anthropic"`` spelled out is the same request as omitting it."""
    from agentlite.testing import MockClient, text_response

    implicit = MockClient(responses=[text_response(response)])
    fn(*args, client=implicit)

    explicit = MockClient(responses=[text_response(response)])
    fn(*args, client=explicit, provider="anthropic")

    assert explicit.last_create_kwargs == implicit.last_create_kwargs


# --------------------------------------------------------------------------- #
# 2. Every agent can reach an OpenAI-compatible endpoint                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("name", "fn", "args", "response"), AGENT_CASES, ids=AGENT_IDS)
def test_agent_runs_against_an_openai_compatible_endpoint(
    name: str,
    fn: Any,
    args: tuple[Any, ...],
    response: str,
) -> None:
    """The agent produces its normal parsed dict over chat completions."""
    client = FakeOpenAIClient(response)
    out = fn(*args, client=client, provider="openai", model="qwen2.5:7b")

    assert isinstance(out, dict)
    assert out  # the agent's own required keys were parsed and returned

    call = client.last_call
    assert call["model"] == "qwen2.5:7b"
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert call["messages"][0]["content"]  # the agent's system prompt
    # Requirement 4: these agents make no tool calls, so the OpenAI path must
    # not bind tools.
    assert "tools" not in call
    assert "tool_choice" not in call


@pytest.mark.parametrize(("name", "fn", "args", "response"), AGENT_CASES, ids=AGENT_IDS)
def test_openai_path_asks_for_a_json_object(
    name: str,
    fn: Any,
    args: tuple[Any, ...],
    response: str,
) -> None:
    client = FakeOpenAIClient(response)
    fn(*args, client=client, provider="openai", model="qwen2.5:7b")
    assert client.last_call["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(("name", "fn", "args", "response"), AGENT_CASES, ids=AGENT_IDS)
def test_openai_path_still_validates_the_response_shape(
    name: str,
    fn: Any,
    args: tuple[Any, ...],
    response: str,
) -> None:
    """A non-JSON reply from an OpenAI endpoint raises the agent's own error."""
    client = FakeOpenAIClient("I am not JSON.")
    with pytest.raises(ValueError):
        fn(*args, client=client, provider="openai", model="qwen2.5:7b")


def test_openai_path_retries_without_an_unsupported_parameter() -> None:
    """Endpoints that reject ``response_format`` still work (one retry)."""
    payload = AGENT_CASES[0][3]
    client = FakeOpenAIClient(
        payload,
        reject_once="Unsupported parameter: 'response_format' is not supported",
    )
    out = get_recommendation(SAMPLE_ANALYSIS, client=client, provider="openai", model="m")

    assert "models" in out
    assert len(client.calls) == 2
    assert "response_format" in client.calls[0]
    assert "response_format" not in client.calls[1]


def test_openai_path_does_not_swallow_a_real_error() -> None:
    """An unrelated failure propagates instead of being retried away."""
    client = FakeOpenAIClient("{}", reject_once="connection refused")
    with pytest.raises(RuntimeError, match="connection refused"):
        get_recommendation(SAMPLE_ANALYSIS, client=client, provider="openai", model="m")
    assert len(client.calls) == 1


# --------------------------------------------------------------------------- #
# 3. Resolution order: argument > environment > built-in default              #
# --------------------------------------------------------------------------- #


def test_resolution_falls_back_to_the_builtin_default() -> None:
    cfg = _common.resolve_llm_config(default_model=ADVISOR_MODEL_DEFAULT)
    assert cfg.provider == "anthropic"
    assert cfg.model == ADVISOR_MODEL_DEFAULT
    assert cfg.base_url is None


def test_environment_beats_the_builtin_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLCOMPASS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MLCOMPASS_LLM_MODEL", "qwen3:4b")
    monkeypatch.setenv("MLCOMPASS_LLM_BASE_URL", "http://localhost:11434/v1")

    cfg = _common.resolve_llm_config(default_model=ADVISOR_MODEL_DEFAULT)
    assert cfg.provider == "openai"
    assert cfg.model == "qwen3:4b"
    assert cfg.base_url == "http://localhost:11434/v1"


def test_explicit_arguments_beat_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLCOMPASS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MLCOMPASS_LLM_MODEL", "qwen3:4b")
    monkeypatch.setenv("MLCOMPASS_LLM_BASE_URL", "http://localhost:11434/v1")

    cfg = _common.resolve_llm_config(
        default_model=ADVISOR_MODEL_DEFAULT,
        provider="anthropic",
        model="claude-haiku-4-5",
        base_url="https://example.invalid",
    )
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-haiku-4-5"
    assert cfg.base_url == "https://example.invalid"


def test_api_key_comes_from_the_providers_own_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    openai_cfg = _common.resolve_llm_config(default_model="m", provider="openai", model="gpt-x")
    assert openai_cfg.api_key == "sk-openai-test"
    assert _common.resolve_llm_config(default_model="m", provider="anthropic").api_key == (
        "sk-ant-test"
    )


def test_local_endpoint_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A keyless local server (ollama, vLLM) is a complete configuration."""
    cfg = _common.resolve_llm_config(
        default_model="m",
        provider="openai",
        model="qwen2.5:7b",
        base_url="http://localhost:11434/v1",
    )
    assert cfg.api_key  # a placeholder, so the SDK does not refuse to construct
    assert _common.is_configured(cfg) is True


def test_unconfigured_openai_endpoint_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key and no base URL against api.openai.com is not configured."""
    cfg = _common.resolve_llm_config(default_model="m", provider="openai", model="gpt-x")
    assert _common.is_configured(cfg) is False


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="provider"):
        _common.resolve_llm_config(default_model="m", provider="bedrock")


def test_non_anthropic_provider_requires_an_explicit_model() -> None:
    """Sending 'claude-opus-4-7' to an OpenAI endpoint would be nonsense."""
    with pytest.raises(ValueError, match="model"):
        _common.resolve_llm_config(default_model=ADVISOR_MODEL_DEFAULT, provider="openai")


@pytest.mark.parametrize(("name", "fn", "args", "response"), AGENT_CASES, ids=AGENT_IDS)
def test_environment_switches_the_agent_to_openai(
    name: str,
    fn: Any,
    args: tuple[Any, ...],
    response: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MLCOMPASS_LLM_PROVIDER=openai`` routes without touching the call site."""
    monkeypatch.setenv("MLCOMPASS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MLCOMPASS_LLM_MODEL", "mistral:7b")

    client = FakeOpenAIClient(response)
    fn(*args, client=client)

    assert client.last_call["model"] == "mistral:7b"


# --------------------------------------------------------------------------- #
# 4. openai is imported lazily                                                #
# --------------------------------------------------------------------------- #


def test_openai_is_not_imported_by_the_anthropic_path() -> None:
    """Requirement 6: an Anthropic-only install must not need ``openai``."""
    code = (
        "import sys;"
        "import mlcompass.cli;"
        "from mlcompass.agents.advise import get_recommendation;"
        "print('openai' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"


# --------------------------------------------------------------------------- #
# 5. An empty reply says so                                                   #
# --------------------------------------------------------------------------- #


def test_empty_reply_is_reported_as_empty() -> None:
    """Reasoning models sometimes return nothing after thinking.

    Observed once against a local qwen3:4b: the endpoint returned
    ``finish_reason="stop"`` with empty content. The old message,
    "response was not valid JSON: ''", blamed the JSON; the honest message
    is that there was no response at all.
    """
    with pytest.raises(_common.AgentResponseError, match="empty"):
        _common.parse_json_response("", required_keys=("a",))
    with pytest.raises(_common.AgentResponseError, match="empty"):
        _common.parse_json_response("   \n ", required_keys=("a",))


def test_empty_openai_reply_raises_the_agents_own_error() -> None:
    """The CLI catches the agent's error class, so it must be raised."""
    from mlcompass.agents.advise import AdvisorParseError

    client = FakeOpenAIClient("")
    with pytest.raises(AdvisorParseError, match="empty"):
        get_recommendation(SAMPLE_ANALYSIS, client=client, provider="openai", model="m")
