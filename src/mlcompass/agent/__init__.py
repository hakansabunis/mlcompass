"""Self-driving agent layer — Faz 7.

Exposes mlcompass's eight tools to an LLM through one of two backends:

- ``anthropic_api`` (default) — universal, only needs ``ANTHROPIC_API_KEY``.
- ``claude_code`` (opt-in) — requires the Claude Code CLI to be installed
  on the machine; routes through ``claude-agent-sdk``.

The public entry point is :func:`mlcompass.agent.orchestrator.run_agent`.
The CLI wrapper is :command:`mlcompass agent "<task>"`.
"""

from __future__ import annotations
