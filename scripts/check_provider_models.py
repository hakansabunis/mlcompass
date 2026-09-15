"""Probe candidate models for the two capabilities the benchmark needs.

Listing a model does not mean you can call it, and calling it does not mean
it can hold a tool schema. Both matter here for different arms:

- the **A/B benchmark** needs only chat, so a small local model qualifies;
- the **contract arm** needs JSON-schema tool calling, because Tier A binds
  enum domains into a tool input and Tier B reads the returned call. A model
  that cannot emit a well-formed tool call cannot run that arm at all.

So each candidate gets two requests: a trivial chat completion, and a tool
call against a two-field schema with an enum. Both are a handful of tokens,
which keeps a full sweep of the paid providers well under a cent.

    python scripts/check_provider_models.py            # every candidate
    python scripts/check_provider_models.py --provider ollama

Keys are read from `api_keys.txt` at the repo root (gitignored) or from the
environment, and are never printed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEY_FILE = ROOT / "api_keys.txt"
TIMEOUT = 90

# The tool a candidate has to produce. Deliberately shaped like the real
# one: an enum-constrained field plus a number, which is where weak
# implementations fail — they emit prose, or a free-text value the enum
# does not allow.
PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "report_column",
        "description": "Report one column and its measured correlation.",
        "parameters": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "enum": ["alpha", "beta", "gamma"]},
                "correlation": {"type": "number"},
            },
            "required": ["column", "correlation"],
            "additionalProperties": False,
        },
    },
}

CANDIDATES: dict[str, dict[str, Any]] = {
    "deepseek": {
        "key": "deepseek_api",
        "base": "https://api.deepseek.com",
        "models": ["deepseek-flash", "deepseek-v4-pro"],
    },
    "openai": {
        "key": "chatgpt_api",
        "base": "https://api.openai.com/v1",
        "models": [
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.4-mini",
            "gpt-5-nano",
            "gpt-4o-mini",
        ],
    },
    "mistral": {
        "key": "mistral_api",
        "base": "https://api.mistral.ai/v1",
        "models": ["mistral-small-latest", "ministral-8b-latest", "ministral-3b-latest"],
    },
    "ollama": {
        "key": None,
        "base": "http://localhost:11434/v1",
        "models": ["qwen2.5:7b", "qwen3:4b", "mistral:7b", "qwen3-vl:4b"],
    },
}


def load_keys() -> dict[str, str]:
    keys: dict[str, str] = {}
    if KEY_FILE.exists():
        for line in KEY_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                name, _, value = line.partition("=")
                keys[name.strip()] = value.strip().strip("\"'")
    return keys


def _post(base: str, key: str | None, payload: dict[str, Any]) -> tuple[bool, Any, float]:
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"{base}/chat/completions", data=body, headers=headers)
    clock = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:  # noqa: S310 - fixed hosts
            return True, json.loads(r.read().decode()), time.monotonic() - clock
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        try:
            detail = json.loads(detail).get("error", {}).get("message", detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        return False, f"HTTP {exc.code}: {str(detail)[:140]}", time.monotonic() - clock
    except Exception as exc:  # noqa: BLE001 - probe reports whatever went wrong
        return False, f"{type(exc).__name__}: {str(exc)[:140]}", time.monotonic() - clock


def probe_chat(base: str, key: str | None, model: str) -> tuple[str, str, int]:
    ok, resp, secs = _post(
        base,
        key,
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
            "max_completion_tokens": 16,
        },
    )
    if not ok:
        # Some endpoints still want the retired parameter name.
        ok, resp, secs = _post(
            base,
            key,
            {
                "model": model,
                "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
                "max_tokens": 16,
            },
        )
    if not ok:
        return "FAIL", str(resp), 0
    text = (resp["choices"][0]["message"].get("content") or "").strip()
    used = (resp.get("usage") or {}).get("total_tokens", 0)
    return ("ok", text[:24] or "(empty)", used) if text else ("EMPTY", "(no content)", used)


def probe_tools(base: str, key: str | None, model: str) -> tuple[str, str]:
    ok, resp, _ = _post(
        base,
        key,
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Call report_column for the column beta, whose measured "
                        "correlation is 0.97."
                    ),
                }
            ],
            "tools": [PROBE_TOOL],
            "tool_choice": "auto",
            "max_completion_tokens": 128,
        },
    )
    # The gpt-5.6 family defaults to a reasoning effort that its
    # chat-completions endpoint refuses to combine with function tools
    # ("Function tools with reasoning_effort are not supported"). The
    # capability is there; the default parameter is what blocks it. Retry
    # once with reasoning switched off before calling the model incapable —
    # the first version of this probe reported luna, sol and terra as
    # tool-incapable purely because of this.
    if not ok and "reasoning_effort" in str(resp):
        ok, resp, _ = _post(
            base,
            key,
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Call report_column for the column beta, whose measured "
                            "correlation is 0.97."
                        ),
                    }
                ],
                "tools": [PROBE_TOOL],
                "tool_choice": "auto",
                "reasoning_effort": "none",
                "max_completion_tokens": 128,
            },
        )
        if ok:
            calls = resp["choices"][0]["message"].get("tool_calls") or []
            if calls:
                try:
                    args = json.loads(calls[0]["function"]["arguments"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    return "BAD-JSON", "needs reasoning_effort=none"
                return "ok*", f"column={args.get('column')} (needs reasoning_effort=none)"

    if not ok:
        return "FAIL", str(resp)
    calls = resp["choices"][0]["message"].get("tool_calls") or []
    if not calls:
        return "NO-CALL", "answered in prose instead of calling the tool"
    try:
        args = json.loads(calls[0]["function"]["arguments"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return "BAD-JSON", f"{type(exc).__name__}"
    if args.get("column") not in ("alpha", "beta", "gamma"):
        return "ENUM-BREAK", f"column={args.get('column')!r} is outside the enum"
    return "ok", f"column={args.get('column')} correlation={args.get('correlation')}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", action="append", choices=sorted(CANDIDATES))
    args = ap.parse_args()

    keys = load_keys()
    providers = args.provider or list(CANDIDATES)
    usable_for_contract: list[str] = []
    usable_for_chat: list[str] = []

    for provider in providers:
        spec = CANDIDATES[provider]
        key = keys.get(spec["key"], "") if spec["key"] else None
        if spec["key"] and not key:
            print(f"\n{provider}: no key found for {spec['key']}; skipped")
            continue

        print(f"\n{provider}  ({spec['base']})")
        print(f"  {'model':<24} {'chat':<8} {'tools':<12} detail")
        print(f"  {'-' * 24} {'-' * 8} {'-' * 12} {'-' * 40}")
        for model in spec["models"]:
            chat_status, chat_detail, tokens = probe_chat(spec["base"], key, model)
            if chat_status == "FAIL":
                print(f"  {model:<24} {'FAIL':<8} {'-':<12} {chat_detail[:60]}")
                continue
            tool_status, tool_detail = probe_tools(spec["base"], key, model)
            label = f"{provider}/{model}"
            usable_for_chat.append(label)
            if tool_status in ("ok", "ok*"):
                usable_for_contract.append(label)
            print(
                f"  {model:<24} {chat_status:<8} {tool_status:<12} "
                f"{tool_detail[:52]}  [{tokens} tok]"
            )

    print(f"\n{'=' * 78}")
    print(f"Chat-capable ({len(usable_for_chat)}) — usable for the with/without A/B benchmark:")
    for m in usable_for_chat:
        print(f"    {m}")
    print(f"\nTool-calling ({len(usable_for_contract)}) — also usable for the contract arm:")
    for m in usable_for_contract:
        print(f"    {m}")
    no_tools = [m for m in usable_for_chat if m not in usable_for_contract]
    if no_tools:
        print(f"\nChat only, cannot run the contract arm ({len(no_tools)}):")
        for m in no_tools:
            print(f"    {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
