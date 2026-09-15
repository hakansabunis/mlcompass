"""Shared helpers for the ``agents`` package.

Two things live here, both shared by the eight pure-JSON reasoner agents
(advise, audit, watch, compare, evaluate, deploy, monitor, optimize):

**The response contract.** Every one of them asks for a single JSON
object, tolerates markdown ``` fences, and requires a small set of
top-level keys. :func:`parse_json_response` implements that once, so each
agent only declares its keys and its error class.

**The provider path.** :func:`run_json_agent` is the single request path.
Anthropic is the default and keeps going through ``agentlite.Agent``
exactly as before; ``provider="openai"`` goes to the Chat Completions API
instead, which covers every OpenAI-compatible endpoint (a local ollama or
vLLM server, DeepSeek, Groq, Gemini's compatibility layer, …). The shape
mirrors ``agents/leakage_investigator.py``, which was already
provider-aware: an injected client wins, otherwise one is built from the
resolved configuration.

These agents make no tool calls, so the OpenAI path asks for a JSON object
via ``response_format`` and never binds tools.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

# --------------------------------------------------------------------------- #
# Provider configuration                                                      #
# --------------------------------------------------------------------------- #

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
SUPPORTED_PROVIDERS = (PROVIDER_ANTHROPIC, PROVIDER_OPENAI)
DEFAULT_PROVIDER = PROVIDER_ANTHROPIC

ENV_PROVIDER = "MLCOMPASS_LLM_PROVIDER"
ENV_BASE_URL = "MLCOMPASS_LLM_BASE_URL"
ENV_MODEL = "MLCOMPASS_LLM_MODEL"

#: Each provider reads its own conventional key variable, so an existing
#: ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` export just works.
PROVIDER_KEY_ENV = {
    PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
    PROVIDER_OPENAI: "OPENAI_API_KEY",
}

#: Local servers (ollama, vLLM, llama.cpp) authenticate nobody but the SDK
#: still refuses to construct without *some* key. Substituted only when a
#: base URL was given — pointing at api.openai.com with no key should still
#: produce the SDK's own "set OPENAI_API_KEY" error, not a confusing 401.
PLACEHOLDER_API_KEY = "mlcompass-local"

#: Matches ``agentlite.Agent``'s hard-coded cap, so the two paths are
#: bounded the same way.
MAX_TOKENS = 4096

#: Parameters the OpenAI path sends but can live without. Some
#: OpenAI-compatible servers reject one of them outright; when the error
#: names the parameter, we drop it and retry once rather than fail.
_OPTIONAL_REQUEST_PARAMS = ("response_format", "max_tokens")


@dataclass(frozen=True)
class LLMConfig:
    """A fully resolved provider configuration."""

    provider: str
    model: str
    base_url: str | None
    api_key: str | None


def resolve_llm_config(
    *,
    default_model: str,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LLMConfig:
    """Resolve where an agent should send its request.

    Precedence, for every field, is **explicit argument → environment →
    built-in default**:

    ===========  ==========================  ==========================
    Field        Environment variable        Built-in default
    ===========  ==========================  ==========================
    provider     ``MLCOMPASS_LLM_PROVIDER``  ``"anthropic"``
    model        ``MLCOMPASS_LLM_MODEL``     ``default_model`` (the
                                             agent's own constant, only
                                             for the Anthropic default)
    base_url     ``MLCOMPASS_LLM_BASE_URL``  None (the SDK's own default)
    api_key      the provider's own variable None
                 (``ANTHROPIC_API_KEY`` /
                 ``OPENAI_API_KEY``)
    ===========  ==========================  ==========================

    Args:
        default_model: The agent's built-in model constant. Used only when
            the provider is Anthropic; it is a Claude model name, so
            silently sending it to an OpenAI-compatible endpoint would be
            wrong and raises instead.
        provider: ``"anthropic"`` or ``"openai"``.
        model: Model identifier for the chosen provider.
        base_url: Endpoint override — this is what points the OpenAI path
            at a local ollama / vLLM server or another vendor.
        api_key: Credential override.

    Returns:
        The resolved :class:`LLMConfig`.

    Raises:
        ValueError: On an unknown provider, or when a non-Anthropic
            provider was selected without a model.
    """
    resolved_provider = (provider or os.environ.get(ENV_PROVIDER) or DEFAULT_PROVIDER).strip()
    if resolved_provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider {resolved_provider!r}. "
            f"Supported: {', '.join(SUPPORTED_PROVIDERS)}."
        )

    resolved_model = model or os.environ.get(ENV_MODEL) or None
    if resolved_model is None:
        if resolved_provider != PROVIDER_ANTHROPIC:
            raise ValueError(
                f"Provider {resolved_provider!r} needs an explicit model: the built-in "
                f"default ({default_model!r}) is an Anthropic model name. Pass --model, "
                f"or set {ENV_MODEL}."
            )
        resolved_model = default_model

    resolved_base_url = base_url or os.environ.get(ENV_BASE_URL) or None

    resolved_key = api_key or os.environ.get(PROVIDER_KEY_ENV[resolved_provider]) or None
    if resolved_key is None and resolved_provider == PROVIDER_OPENAI and resolved_base_url:
        resolved_key = PLACEHOLDER_API_KEY

    return LLMConfig(
        provider=resolved_provider,
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=resolved_key,
    )


def is_configured(config: LLMConfig) -> bool:
    """True when ``config`` has enough to reach a provider.

    A credential, or a base URL that implies a self-hosted endpoint where
    the credential is a formality. This is what lets ``--llm`` run against
    a local ollama server with no key set anywhere.
    """
    return bool(config.api_key or config.base_url)


def build_client(config: LLMConfig) -> Any:
    """Construct a provider client for ``config``.

    Both SDKs are imported lazily so an install that only ever uses one of
    them never pays for — or needs — the other.
    """
    if config.provider == PROVIDER_OPENAI:
        from openai import OpenAI

        kwargs: dict[str, Any] = {}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return OpenAI(**kwargs)

    import anthropic

    kwargs = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return anthropic.Anthropic(**kwargs)


# --------------------------------------------------------------------------- #
# The shared request path                                                     #
# --------------------------------------------------------------------------- #


def run_json_agent(
    *,
    system: str,
    user: str,
    default_model: str,
    client: Any | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    max_turns: int = 2,
) -> str:
    """Send one system+user exchange and return the raw reply text.

    The eight JSON-reasoner agents all funnel through here, so there is one
    provider implementation rather than eight. Argument/environment/default
    precedence is documented on :func:`resolve_llm_config`.

    On the Anthropic path nothing changed: the request still goes through
    ``agentlite.Agent`` with the same system caching and the same 4096-token
    cap, and when no client is injected agentlite builds its own
    ``anthropic.Anthropic()`` from ``ANTHROPIC_API_KEY`` exactly as before.
    A client is constructed here only when the caller actually asked for a
    non-default endpoint or credential.

    On the OpenAI path the request is a plain chat completion asking for a
    JSON object. No tools are bound — these agents make no tool calls.

    Args:
        system: The agent's system prompt.
        user: The rendered user message.
        default_model: The agent's built-in model constant.
        client: Pre-built provider client (tests inject fakes here; the
            measurement harness injects real ones).
        provider: See :func:`resolve_llm_config`.
        model: See :func:`resolve_llm_config`.
        base_url: See :func:`resolve_llm_config`.
        api_key: See :func:`resolve_llm_config`.
        max_turns: Safety cap on the agentlite loop. Single-turn is the
            expected case for every one of these agents.

    Returns:
        The model's raw reply text, for the caller to parse.
    """
    config = resolve_llm_config(
        default_model=default_model,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )

    if config.provider == PROVIDER_OPENAI:
        api = client if client is not None else build_client(config)
        return _complete_openai(api, model=config.model, system=system, user=user)

    # Anthropic: keep the pre-existing agentlite path untouched. Only build
    # a client when the caller pointed us somewhere other than the default.
    if client is None and (base_url or api_key or config.base_url):
        client = build_client(config)
    agent = build_anthropic_agent(
        system=system,
        model=config.model,
        client=client,
        max_turns=max_turns,
    )
    reply: str = agent.run(user)
    return reply


def build_anthropic_agent(
    *,
    system: str,
    model: str,
    client: Any | None = None,
    max_turns: int = 2,
) -> Any:
    """Build the tool-free ``agentlite.Agent`` these agents have always used."""
    from agentlite import Agent

    return Agent(
        model=model,
        system=system,
        tools=[],  # Pure reasoners — no tool calls needed.
        client=client,
        max_turns=max_turns,
    )


def _complete_openai(api: Any, *, model: str, system: str, user: str) -> str:
    """One chat completion against an OpenAI-compatible endpoint.

    Asks for a JSON object because that is the whole contract of these
    agents. If the endpoint rejects one of the optional parameters by name
    — some compatible servers do not implement ``response_format``, and
    some newer models renamed ``max_tokens`` — the named parameters are
    dropped and the call is retried once. Any other failure propagates: a
    connection error must not be quietly retried into a different request.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": MAX_TOKENS,
    }

    try:
        response = api.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 — re-raised unless clearly ours
        unsupported = [p for p in _OPTIONAL_REQUEST_PARAMS if p in str(exc)]
        if not unsupported:
            raise
        for param in unsupported:
            kwargs.pop(param, None)
        response = api.chat.completions.create(**kwargs)

    content = response.choices[0].message.content
    return str(content) if content else ""


class AgentResponseError(ValueError):
    """Base class for agent JSON-response parse failures."""


def parse_json_response(
    text: str,
    *,
    required_keys: tuple[str, ...] = (),
    error_class: type[AgentResponseError] = AgentResponseError,
    snippet_len: int = 200,
) -> dict[str, Any]:
    """Parse an agent's reply as a JSON object.

    Args:
        text: Raw agent output.
        required_keys: Top-level keys that must be present.
        error_class: Specific subclass of :class:`AgentResponseError`
            to raise on failure (lets callers catch theirs by type).
        snippet_len: How much of the raw text to include in the
            error message.

    Returns:
        The parsed object.

    Raises:
        error_class: On any parse / shape / missing-key failure.
    """
    stripped = text.strip()

    # Distinguish "said nothing" from "said something that isn't JSON".
    # Reasoning models occasionally return an empty completion after a long
    # thinking block; blaming the JSON there sends the reader looking for a
    # parse bug that does not exist.
    if not stripped:
        raise error_class("Agent response was empty (the model returned no content).")

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        start = 1
        end = len(lines)
        for i, line in enumerate(lines[1:], 1):
            if line.strip().startswith("```"):
                end = i
                break
        stripped = "\n".join(lines[start:end])

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        snippet = text[:snippet_len].replace("\n", " ")
        raise error_class(f"Agent response was not valid JSON: {snippet!r}") from exc

    if not isinstance(parsed, dict):
        raise error_class(
            f"Agent response was JSON but not an object (got {type(parsed).__name__})"
        )

    for required in required_keys:
        if required not in parsed:
            raise error_class(
                f"Agent response missing required key '{required}'. Got: {sorted(parsed.keys())}"
            )

    return parsed
