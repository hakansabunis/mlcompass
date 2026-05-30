"""Shared helpers for the ``agents`` package.

The three Faz 2.1 LLM layers (audit prioritizer, watch diagnostician,
compare hypothesizer) all share the same loose JSON-response contract:

* The agent is asked to reply with a single JSON object.
* Optional markdown ``` fences are tolerated.
* A small set of top-level keys is required; everything else is ignored.

Keeping the parser here means each agent only has to declare the keys
it cares about and a custom error class.
"""

from __future__ import annotations

import json
from typing import Any


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
