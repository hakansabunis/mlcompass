"""Reproduce the three-layer phantom-column fabrication ablation.

The ablation isolates the contribution of each layer of the evidence-bound
narration contract on a controlled synthetic data-leakage task:

  * Layer 1 — bare prompt, no schema enum, no validation. The raw fabrication
    rate of an unconstrained narrator.
  * Layer 2 — strict prompt, no schema enum, no validation. How much the prompt
    alone buys you.
  * Layer 3 — strict prompt + evidence-bound schema enum + deterministic
    post-validation. This is the SHIPPED contract: live mode calls the exact
    same ``investigate_leakage_bound`` the product uses, over the exact same
    evidence dict ``detect_leakage`` produces. Every cited column is validated
    against the evidence set and out-of-evidence columns are rejected/stripped,
    so the rate of a phantom reaching the user is structurally bounded.

Two modes:

  --mode live   — fires real API calls. Builds a synthetic predictions frame,
                runs the real ``detect_leakage`` to get the evidence dict, and
                samples N narrator responses per layer. Layer 3 goes through the
                shipped ``investigate_leakage_bound``. Pick the provider with
                --provider (see PROVIDERS: anthropic, deepseek, openai, gemini,
                mistral, xai, qwen, groq, vllm) and export its API key env var.
                **This is the mode that produces the paper's Table I** — the
                certified 2026-06-12 battery (paper Tables I-II) used --provider deepseek.

  --mode mock   — ILLUSTRATIVE ONLY, no API calls. A deterministic simulator for
                grading and for users without a key. The per-layer rates are
                placeholders, and Layer 3's 0% here is true *by construction of
                the simulator* — it is NOT a measurement. Do not cite mock-mode
                output as a result; use --mode live for that.

Money-safety controls (Q1 campaign, Phase 0):

  --check-providers   zero-cost wiring table (which keys are set)
  --dry-run           print the run plan + ESTIMATED cost, exit before any call
  --smoke             N=5, layer1+layer3 only — validate a provider for pennies
  --max-cost-usd X    abort before the first call if the estimate exceeds X
  (default on)        incremental JSONL run log under scripts/runs/ — every paid
                      response is flushed to disk immediately; --resume continues
                      an interrupted run without repeating paid calls

Usage::

    # zero-cost: is everything wired?
    python scripts/reproduce_hallucination_ablation.py --check-providers

    # zero-cost: what would this run cost?
    python scripts/reproduce_hallucination_ablation.py --mode live --provider gemini --dry-run

    # pennies: validate a new provider end-to-end
    python scripts/reproduce_hallucination_ablation.py --mode live --provider gemini --smoke

    # real measurement, N=200 per layer (paper Table I)
    python scripts/reproduce_hallucination_ablation.py --mode live --n 200

    # continue an interrupted battery without re-paying for finished calls
    python scripts/reproduce_hallucination_ablation.py --mode live --n 200 --resume

    # no-API illustrative demo (clearly labelled as such)
    python scripts/reproduce_hallucination_ablation.py --mode mock

The ``--json`` flag emits a machine-readable summary for a regression harness.
See ``docs/THREAT_MODEL.md`` and ``docs/KNOWN_LIMITATIONS.md`` for what the
contract does and does not promise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any

# The script measures the SHIPPED contract — import the product code rather
# than reimplementing it, so the experiment can never drift from the artifact.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
# Sibling module (leak injectors); needed when the harness is loaded via
# importlib (tests) rather than executed as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fabbench_injectors import list_injectors  # noqa: E402

from mlcompass.agents.leakage_investigator import (  # noqa: E402
    LEAKAGE_BOUND_PROMPT,
    SUBMIT_TOOL_NAME,
    VALUE_TOLERANCE,
    build_submit_investigation_tool,
    build_submit_investigation_tool_openai,
    evidence_allowed_columns,
    evidence_correlation_map,
    investigate_leakage_bound,
    top_candidate,
)

# --------------------------------------------------------------------------- #
# Wilson 95% confidence interval                                              #
# --------------------------------------------------------------------------- #


def wilson_ci_95(k: int, n: int) -> tuple[float, float]:
    """Return the (low, high) bounds of the Wilson 95% CI for k/n.

    We use the score-interval formulation (no continuity correction) because
    it produces sensible bounds when k is 0 or n, which the normal
    approximation does not.

    References
    ----------
    Wilson, E. B. (1927). Probable inference, the law of succession, and
    statistical inference. JASA, 22(158), 209-212.
    """
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # 95% two-sided normal quantile
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (low, high)


# --------------------------------------------------------------------------- #
# Synthetic evidence — built by the REAL detector                             #
# --------------------------------------------------------------------------- #


def build_synthetic_evidence(seed: int = 0) -> dict[str, Any]:
    """Construct a synthetic predictions frame with a known transformed leak
    and run the real ``detect_leakage`` on it.

    Using the shipped detector means the evidence dict the experiment narrates
    is byte-identical in shape to what the product surfaces — closing the
    "experiment evidence != product evidence" gap a reviewer flagged.
    """
    import numpy as np
    import pandas as pd

    from mlcompass.tools.leakage import detect_leakage

    rng = np.random.default_rng(seed)
    n = 1000
    y = rng.normal(50.0, 10.0, n)
    df = pd.DataFrame({"y_true": y})
    # A monotone (log) transform of the target — Pearson ~0.94, Spearman 1.0.
    df["log_target_v2"] = np.log(y - y.min() + 1.0) + rng.normal(0, 0.01, n)
    # A noisy near-copy of the target — a high-correlation distractor.
    df["near_target_proxy"] = y + rng.normal(0, 0.5, n)
    for i in range(3, 13):
        df[f"feature_{i}"] = rng.normal(0, 1, n)
    # A model that essentially copies the leak → near-perfect predictions.
    df["y_pred"] = y + rng.normal(0, 0.001, n)

    return detect_leakage(
        df,
        y_true_col="y_true",
        y_pred_col="y_pred",
        task="regression",
        suspicious_metric={"name": "r2", "value": 1.0},
    )


def build_csv_evidence(
    csv_path: str, target: str, seed: int = 0, injector: str = "monotone_log"
) -> dict[str, Any]:
    """Build evidence from a REAL third-party dataset with an injected leak.

    Loads the CSV, plants ONE controlled leakage pattern (``--injector``, see
    ``fabbench_injectors``) plus near-perfect predictions, and runs the
    shipped ``detect_leakage``. The default injector is the June 2026
    ``monotone_log`` pattern, so pre-Phase-2 commands reproduce byte-identical
    task setups. The feature distributions, column names, and dataset shape
    are all external — answering the "everything is self-designed" critique
    with real-data replication tasks.
    """
    import numpy as np
    import pandas as pd
    from fabbench_injectors import inject

    from mlcompass.tools.leakage import detect_leakage

    rng = np.random.default_rng(seed)
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise SystemExit(
            f"--target '{target}' not found in {csv_path}. Columns: {list(df.columns)}"
        )
    y = pd.to_numeric(df[target], errors="coerce")
    if y.isna().all():
        raise SystemExit(f"--target '{target}' is not numeric in {csv_path}.")
    df = df.loc[y.notna()].reset_index(drop=True)
    y_arr = y.dropna().to_numpy(dtype=float)
    n = len(df)
    df["y_pred"] = y_arr + rng.normal(0, max(1e-9, 0.001 * y_arr.std()), n)
    inject(df, y_arr, target, injector, rng)

    return detect_leakage(
        df,
        y_true_col=target,
        y_pred_col="y_pred",
        task="regression",
        suspicious_metric={"name": "r2", "value": 1.0},
    )


# --------------------------------------------------------------------------- #
# Mock backend — ILLUSTRATIVE ONLY (no API, not a measurement)                #
# --------------------------------------------------------------------------- #


# Per-layer illustrative rates for the no-API demo, set to the 2026-06-12
# deepseek-chat battery run (see paper/ablation_live_deepseek_
# 2026-06-12_battery.md). Mock mode REPLAYS rates; it does not measure.
ILLUSTRATIVE_RATES: dict[str, dict[str, float]] = {
    "layer1": {"entity": 0.115, "value": 0.000, "omission": 0.000},
    "layer2": {"entity": 0.000, "value": 0.000, "omission": 0.000},
    "layer3": {"entity": 0.000, "value": 0.000, "omission": 0.000},
}

PLAUSIBLE_PHANTOMS = (
    "target_score",
    "predicted_value",
    "leak_indicator",
    "label_proxy",
    "revenue",
    "shadow_target",
)

_LAYER_SEED_OFFSET = {"layer1": 1, "layer2": 2, "layer3": 3}


def _mock_one_response(
    rng: random.Random,
    layer: str,
    allowed: list[str],
    corr_map: dict[str, float],
    anchor: str | None,
) -> dict[str, Any]:
    rates = ILLUSTRATIVE_RATES[layer]
    cited = [anchor or rng.choice(allowed)]
    for _ in range(rng.randint(0, 2)):
        cited.append(rng.choice(allowed))
    claims: list[dict[str, Any]] = []
    if corr_map:
        col = cited[0] if cited[0] in corr_map else next(iter(corr_map))
        true_val = corr_map[col]
        # Value channel: misquote with the layer's illustrative probability.
        val = true_val if rng.random() >= rates["value"] else round(true_val - 0.15, 2)
        claims.append({"column": col, "statistic": "correlation", "value": val})
    # Layer 3's simulator never fabricates — the "by construction" property
    # the live mode has to demonstrate empirically.
    if layer != "layer3":
        if rng.random() < rates["entity"]:
            cited.append(rng.choice(PLAUSIBLE_PHANTOMS))
        if anchor is not None and rng.random() < rates["omission"]:
            cited = [c for c in cited if c != anchor]
            claims = [c for c in claims if c.get("column") != anchor]
    return {"columns": cited, "claims": claims, "verdict": "leakage_likely", "omitted": None}


def mock_run(
    n: int, allowed: list[str], corr_map: dict[str, float], anchor: str | None, seed: int = 0
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for layer in ILLUSTRATIVE_RATES:
        rng = random.Random(seed * 100003 + _LAYER_SEED_OFFSET[layer])
        out[layer] = [_mock_one_response(rng, layer, allowed, corr_map, anchor) for _ in range(n)]
    return out


# --------------------------------------------------------------------------- #
# Live backend — measures the SHIPPED contract                                #
# --------------------------------------------------------------------------- #


LIVE_SYSTEM_PROMPT_BARE = """You are mlcompass-leakage-investigator.

A deterministic tool gathered evidence about a suspicious metric on a
predictions table. Call the submit_investigation tool once. In
columns_referenced, list every column from the evidence you cite. Report each
correlation number you mention in claims as {"column": ..., "statistic":
"correlation", "value": ...}. Include a verdict and a short narration.
"""

# Six bare-prompt paraphrases for the --sweep mode. All state the same task
# and the same output fields; none states a faithfulness rule. The variation
# is purely stylistic — the sweep measures how much the entity-fabrication
# rate moves under innocuous rewording, with the tool schema held constant.
SWEEP_BARE_VARIANTS: list[tuple[str, str]] = [
    (
        "terse",
        (
            "A tool produced leakage evidence for a suspicious metric. Call "
            "submit_investigation once: columns_referenced, claims "
            '({"column","statistic":"correlation","value"}), a verdict, a short narration.'
        ),
    ),
    ("baseline", LIVE_SYSTEM_PROMPT_BARE),
    (
        "helpful",
        (
            "You are a friendly ML assistant helping an engineer understand a "
            "suspiciously perfect metric. A deterministic tool collected evidence "
            "about the predictions table. Explain what is going on for them. Submit "
            "your analysis through the submit_investigation tool: list the columns "
            "you discuss in columns_referenced, put every correlation number you "
            "mention into claims, give a verdict and a clear narration."
        ),
    ),
    (
        "expert",
        (
            "You are a senior machine-learning engineer performing a leakage "
            "review. The deterministic evidence for a suspicious metric is below. "
            "Deliver your professional assessment via the submit_investigation "
            "tool, filling columns_referenced, claims, verdict, and narration."
        ),
    ),
    (
        "mechanical",
        (
            "Task: leakage investigation. Input: evidence dictionary. Output: one "
            "submit_investigation call with fields columns_referenced (array), "
            "claims (array of column/statistic/value), verdict (string), narration "
            "(string)."
        ),
    ),
    (
        "cautious",
        (
            "You are an ML diagnostics assistant. A deterministic tool gathered "
            "evidence about a suspicious metric. Analyse it carefully and be "
            "accurate. Answer once through the submit_investigation tool with "
            "columns_referenced, claims for each correlation you mention, a "
            "verdict, and a short narration."
        ),
    ),
]


# Provider table: which API key, base URL, and default model each uses.
# Default models are the Q1-campaign SUBJECT tier fixed in
# paper/Q1_ROADMAP.md §2b (verified 2026-07-07); override with --model.
# All non-Anthropic providers speak the OpenAI-compatible Chat Completions
# protocol, so one client path covers them via base_url. Enforcement of
# schema enums varies by provider and mode (always-strict / opt-in / hints);
# Tier B never relies on it — that is the provider-independence claim.
PROVIDERS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "kind": "anthropic",
        "key_env": "ANTHROPIC_API_KEY",
        # Panel subject tier (product default is LEAKAGE_MODEL_DEFAULT).
        "model": "claude-haiku-4-5",
        "base_url": None,
    },
    "deepseek": {
        "kind": "openai",
        "key_env": "DEEPSEEK_API_KEY",
        # The 'deepseek-chat' alias is deprecated 2026-07-24; pin the direct
        # name so replication runs after that date keep working.
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
    },
    "openai": {
        "kind": "openai",
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-5.4-mini",
        "base_url": None,
    },
    "gemini": {
        "kind": "openai",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash-lite",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "mistral": {
        "kind": "openai",
        "key_env": "MISTRAL_API_KEY",
        "model": "mistral-small-latest",
        "base_url": "https://api.mistral.ai/v1",
    },
    "xai": {
        "kind": "openai",
        "key_env": "XAI_API_KEY",
        # xAI enforces tool schemas unconditionally ("strict implicitly always
        # true") — the panel's always-enforced (E-class) data point.
        "model": "grok-4.20-0309-non-reasoning",
        "base_url": "https://api.x.ai/v1",
    },
    "qwen": {
        "kind": "openai",
        "key_env": "DASHSCOPE_API_KEY",
        # DashScope international (Singapore) OpenAI-compatible mode.
        "model": "qwen-flash",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    },
    "groq": {
        "kind": "openai",
        "key_env": "GROQ_API_KEY",
        "model": "llama-3.1-8b-instant",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "vllm": {
        "kind": "openai",
        "key_env": "VLLM_API_KEY",
        # Local vLLM OpenAI-compatible server (WSL2); the decode-enforced
        # Outlines/GCD baseline lane. Server ignores the key ("EMPTY" ok).
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "key_optional": True,
    },
}


# --------------------------------------------------------------------------- #
# Cost estimation (--dry-run) — burns zero dollars                            #
# --------------------------------------------------------------------------- #

# $ per 1M tokens (input, output). Verified against official pricing pages on
# 2026-07-07 — see paper/Q1_ROADMAP.md §2b for sources. Estimates only; the
# authoritative spend is whatever the provider bills.
PRICING_PER_M: dict[str, tuple[float, float]] = {
    "deepseek-v4-flash": (0.14, 0.28),
    "deepseek-chat": (0.14, 0.28),  # legacy alias of deepseek-v4-flash
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "claude-haiku-4-5": (1.00, 5.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "mistral-small-latest": (0.15, 0.60),
    "ministral-3b-latest": (0.10, 0.10),
    "grok-4.20-0309-non-reasoning": (1.25, 2.50),
    "qwen-flash": (0.05, 0.40),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "Qwen/Qwen2.5-3B-Instruct": (0.0, 0.0),  # local vLLM
    # Strong tier (P2 spot-checks / judges):
    "gpt-5.5": (5.00, 30.00),
    "claude-opus-4-8": (5.00, 25.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),
}

# Rough per-call token estimate from the June 2026 battery runs: the evidence
# JSON plus prompt lands around ~1.3k input tokens; a tool-call answer around
# ~300 output tokens. Deliberately round; used for the --dry-run banner only.
EST_IN_TOKENS_PER_CALL = 1300
EST_OUT_TOKENS_PER_CALL = 300

# Expected provider calls per response, by arm: contract arms may issue up to
# two corrective retries. Factors follow the measured June rates (production
# ~1.0; stress on the hard task ~1.45).
ARM_CALL_FACTOR: dict[str, float] = {
    "layer1": 1.0,
    "layer2": 1.0,
    "layer3": 1.05,
    "layer3_stress": 1.45,
    "tier_a": 1.0,
    "stress_mech": 1.05,
}


def estimate_cost_usd(model: str, arms: tuple[str, ...], n: int) -> tuple[float | None, int]:
    """Return (estimated USD or None if pricing unknown, estimated call count)."""
    calls = sum(int(math.ceil(n * ARM_CALL_FACTOR.get(arm, 1.0))) for arm in arms)
    pricing = PRICING_PER_M.get(model)
    if pricing is None:
        return None, calls
    p_in, p_out = pricing
    usd = calls * (EST_IN_TOKENS_PER_CALL * p_in + EST_OUT_TOKENS_PER_CALL * p_out) / 1_000_000.0
    return usd, calls


# Open (unenforced) schema for layers 1 and 2: same fields as the bound tool,
# but no enums and no post-validation, so raw fabrication is observable on all
# three channels (entities, values, omissions).
_OPEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string"},
        "confidence": {"type": "string"},
        "columns_referenced": {"type": "array", "items": {"type": "string"}},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "statistic": {"type": "string"},
                    "value": {"type": "number"},
                },
                "required": ["column", "statistic", "value"],
            },
        },
        "narration": {"type": "string"},
    },
    "required": ["verdict", "columns_referenced", "claims", "narration"],
}


def _open_submit_tool_anthropic() -> dict[str, Any]:
    """Anthropic submit tool WITHOUT the evidence enum (layers 1 and 2)."""
    return {
        "name": SUBMIT_TOOL_NAME,
        "description": "Submit the leakage investigation result.",
        "input_schema": _OPEN_SCHEMA,
    }


def _open_submit_tool_openai() -> dict[str, Any]:
    """OpenAI-compatible submit tool WITHOUT the evidence enum (layers 1 and 2)."""
    return {
        "type": "function",
        "function": {
            "name": SUBMIT_TOOL_NAME,
            "description": "Submit the leakage investigation result.",
            "parameters": _OPEN_SCHEMA,
        },
    }


def _user_message(evidence: dict[str, Any]) -> str:
    return (
        "Evidence dictionary:\n\n"
        + json.dumps(evidence, indent=2, default=str)
        + "\n\nInvestigate the suspicious metric."
    )


# A scored response is a normalized dict:
#   {"columns": [...], "claims": [{column, statistic, value}, ...], "verdict": str}
# For layer 3 it is the USER-FACING (already validated/stripped) result, plus
# the contract's omission flag carried through as "omitted".


def _from_contract_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize an ``investigate_leakage_bound`` result for scoring/logging.

    Provider token usage is not surfaced by the contract path (it may issue
    up to three internal calls), so usage stays None there; latency is set by
    the run loop around the whole contract call.
    """
    out = _normalize({})
    out.update(
        {
            "columns": list(result["columns_referenced"]),
            "claims": list(result["claims"]),
            "verdict": result["verdict"],
            "omitted": bool(result["omitted_critical_evidence"]),
            "rejections": int(result["schema_rejections"]),
            "rejection_kinds": list(result.get("rejection_kinds") or []),
        }
    )
    return out


def _normalize(tool_input: dict[str, Any]) -> dict[str, Any]:
    cols = tool_input.get("columns_referenced") or []
    claims = [c for c in (tool_input.get("claims") or []) if isinstance(c, dict)]
    return {
        "columns": [str(c) for c in cols] if isinstance(cols, list) else [],
        "claims": claims,
        "verdict": str(tool_input.get("verdict", "")),
        "omitted": None,  # computed by the scorer for layers 1-2
        "rejections": 0,  # Tier B catches; nonzero only on contract arms
        "rejection_kinds": [],  # violation composition per catch (contract arms)
        "usage_in": None,  # prompt tokens, when the provider reports them
        "usage_out": None,  # completion tokens, when the provider reports them
        "latency_ms": None,  # wall-clock per response; set by the run loop
    }


def _attach_usage_anthropic(normalized: dict[str, Any], response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is not None:
        normalized["usage_in"] = getattr(usage, "input_tokens", None)
        normalized["usage_out"] = getattr(usage, "output_tokens", None)
    return normalized


def _attach_usage_openai(normalized: dict[str, Any], response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is not None:
        normalized["usage_in"] = getattr(usage, "prompt_tokens", None)
        normalized["usage_out"] = getattr(usage, "completion_tokens", None)
    return normalized


def _input_from_anthropic(response: Any) -> dict[str, Any]:
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            raw = getattr(block, "input", {})
            return _attach_usage_anthropic(
                _normalize(dict(raw) if isinstance(raw, dict) else {}), response
            )
    return _attach_usage_anthropic(_normalize({}), response)


def _input_from_openai(response: Any) -> dict[str, Any]:
    message = response.choices[0].message
    calls = getattr(message, "tool_calls", None) or []
    if not calls:
        return _attach_usage_openai(_normalize({}), response)
    args = calls[0].function.arguments
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
    except (json.JSONDecodeError, TypeError):
        return _attach_usage_openai(_normalize({}), response)
    return _attach_usage_openai(_normalize(parsed if isinstance(parsed, dict) else {}), response)


def _live_one_response(
    kind: str, client: Any, model: str, layer: str, evidence: dict[str, Any], allowed: list[str]
) -> dict[str, Any]:
    """Sample one narrator response for the given layer and provider kind.

    Layers 1 and 2 use an open tool (no enum) with no post-validation, so the
    raw fabrication is observable on all three channels. Layer 3 routes through
    the SHIPPED ``investigate_leakage_bound`` — the same code the product runs —
    and the returned dict is the user-facing result after Tier B validation.
    The stress arm (``layer3_stress``) is the diagnostic configuration that
    demonstrates Tier B catching live violations on its own: it runs the same
    shipped contract but with the BARE prompt and WITHOUT the Tier A enums, so
    the narrator fabricates at its natural rate and every catch is visible in
    ``schema_rejections``.
    """
    if layer == "tier_a":
        # Tier A in isolation: the evidence-bound enum is IN the schema, the
        # bare prompt states no rules, and Tier B verification is NOT applied.
        # Measures how much the runtime enum alone steers this endpoint.
        tool = (
            build_submit_investigation_tool_openai(allowed)
            if kind == "openai"
            else build_submit_investigation_tool(allowed)
        )
        for attempt in range(2):
            try:
                if kind == "openai":
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": LIVE_SYSTEM_PROMPT_BARE},
                            {"role": "user", "content": _user_message(evidence)},
                        ],
                        tools=[tool],
                        tool_choice="required",
                    )
                    return _input_from_openai(response)
                response = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=LIVE_SYSTEM_PROMPT_BARE,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": SUBMIT_TOOL_NAME},
                    messages=[{"role": "user", "content": _user_message(evidence)}],
                )
                return _input_from_anthropic(response)
            except Exception:  # noqa: BLE001
                if attempt == 0:
                    continue
                return _normalize({})
        return _normalize({})

    if layer == "stress_mech":
        # The FULL shipped contract (Tier A enums ON + Tier B) under the worst
        # naturally occurring paraphrase from the sweep (100% bare fabrication)
        # — answers "necessity shown only in an artificially weakened config".
        mech_prompt = dict(SWEEP_BARE_VARIANTS)["mechanical"]
        # Same transient-error policy as the open arms: a 429/timeout must
        # not kill a multi-cell battery mid-run (in-flight money + all
        # remaining cells); after two failures record an empty response.
        for attempt in range(2):
            try:
                result = investigate_leakage_bound(
                    evidence,
                    client=client,
                    model=model,
                    provider=("openai" if kind == "openai" else "anthropic"),
                    system_prompt=mech_prompt,
                )
                return _from_contract_result(result)
            except Exception:  # noqa: BLE001 — SDK exception types vary
                if attempt == 0:
                    continue
        return _normalize({})

    if layer in ("layer3", "layer3_stress"):
        stress = layer == "layer3_stress"
        for attempt in range(2):
            try:
                result = investigate_leakage_bound(
                    evidence,
                    client=client,
                    model=model,
                    provider=("openai" if kind == "openai" else "anthropic"),
                    system_prompt=(LIVE_SYSTEM_PROMPT_BARE if stress else None),
                    enforce_schema_enum=not stress,
                )
                return _from_contract_result(result)
            except Exception:  # noqa: BLE001 — SDK exception types vary
                if attempt == 0:
                    continue
        return _normalize({})

    system = LIVE_SYSTEM_PROMPT_BARE if layer == "layer1" else LEAKAGE_BOUND_PROMPT
    for attempt in range(2):
        try:
            if kind == "openai":
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": _user_message(evidence)},
                    ],
                    tools=[_open_submit_tool_openai()],
                    tool_choice="required",
                )
                return _input_from_openai(response)
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=system,
                tools=[_open_submit_tool_anthropic()],
                tool_choice={"type": "tool", "name": SUBMIT_TOOL_NAME},
                messages=[{"role": "user", "content": _user_message(evidence)}],
            )
            return _input_from_anthropic(response)
        except Exception:  # noqa: BLE001 — SDK exception types vary
            if attempt == 0:
                continue
            return _normalize({})
    return _normalize({})


def _build_live_client(provider: str) -> tuple[str, Any, str]:
    """Construct the provider client. Returns (kind, client, model)."""
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["key_env"])
    if not key:
        if cfg.get("key_optional"):
            key = "EMPTY"  # local servers (vLLM) accept any placeholder key
        else:
            raise SystemExit(
                f"Live mode with provider '{provider}' requires {cfg['key_env']} to be set."
            )
    if cfg["kind"] == "openai":
        try:
            from openai import OpenAI
        except ImportError as e:
            raise SystemExit("Live mode with this provider requires 'pip install openai'.") from e
        client = (
            OpenAI(api_key=key, base_url=cfg["base_url"])
            if cfg["base_url"]
            else OpenAI(api_key=key)
        )
    else:
        try:
            import anthropic
        except ImportError as e:
            raise SystemExit("Live mode with anthropic requires 'pip install anthropic'.") from e
        client = anthropic.Anthropic(api_key=key)
    return cfg["kind"], client, cfg["model"]


# --------------------------------------------------------------------------- #
# Incremental run log — paid responses survive crashes and are resumable      #
# --------------------------------------------------------------------------- #


def _sanitize(part: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in part)


def _evidence_hash(evidence: dict[str, Any]) -> str:
    """Short content hash of the evidence dict — the run cell's true identity."""
    canonical = json.dumps(evidence, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


class RunLog:
    """Append-only JSONL log of live responses, keyed by the run cell.

    Every paid response is flushed to disk the moment it arrives, so a crash
    or Ctrl-C never loses money. With ``resume=True`` an existing log is
    reloaded and the loop continues from where it stopped — the deterministic
    filename (provider/model/task/arm/n/seed) identifies the cell.
    """

    def __init__(
        self,
        log_dir: str,
        provider: str,
        model: str,
        task_label: str,
        arm: str,
        n: int,
        seed: int,
        resume: bool,
        evidence_hash: str = "",
    ) -> None:
        os.makedirs(log_dir, exist_ok=True)
        # The evidence hash ties the cell to the EXACT evidence dict, so a
        # resume against a renamed/modified CSV (same basename) cannot pool
        # responses narrated from different evidence into one cell.
        ev = f"_e{evidence_hash}" if evidence_hash else ""
        name = (
            f"{_sanitize(provider)}_{_sanitize(model)}_{_sanitize(task_label)}"
            f"_{_sanitize(arm)}_n{n}_seed{seed}{ev}.jsonl"
        )
        self.path = os.path.join(log_dir, name)
        self.prior: list[dict[str, Any]] = []
        if resume and os.path.exists(self.path):
            bad_tail = False
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.prior.append(json.loads(line))
                    except json.JSONDecodeError:
                        bad_tail = True  # truncated tail from a crash
                        break
            if bad_tail:
                # Repair BEFORE the first append: rewrite the good prefix so
                # the next record cannot fuse onto the partial line (which
                # would corrupt the log and cause later resumes to re-pay
                # for every record after the corruption point).
                with open(self.path, "w", encoding="utf-8") as f:
                    for record in self.prior:
                        f.write(json.dumps(record, default=str) + "\n")
                print(
                    f"  resume: repaired a truncated tail in {os.path.basename(self.path)}",
                    file=sys.stderr,
                )
            if self.prior:
                print(
                    f"  resume: {len(self.prior)} logged responses found in "
                    f"{os.path.basename(self.path)}",
                    file=sys.stderr,
                )
        elif os.path.exists(self.path):
            raise SystemExit(
                f"Run log already exists: {self.path}\n"
                "Pass --resume to continue it, or move it aside. Refusing to "
                "silently overwrite paid responses."
            )

    def append(self, record: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()


def _run_cell(
    model: str,
    n: int,
    scorer_ctx: tuple[set[str], dict[str, float], str | None],
    log: RunLog | None,
    label: str,
    one: Any,
) -> list[dict[str, Any]]:
    """Run one cell (arm or sweep variant) with logging, resume, and latency."""
    import time

    allowed_set, corr_map, anchor = scorer_ctx
    responses: list[dict[str, Any]] = list(log.prior) if log else []
    start = len(responses)
    if start >= n:
        print(f"  {label}: already complete ({start}/{n}), skipping", file=sys.stderr)
        return responses[:n]
    for i in range(start, n):
        t0 = time.perf_counter()
        r = one()
        r["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        responses.append(r)
        if log:
            flags = score_one(r, allowed_set, corr_map, anchor)
            log.append({"i": i, "arm": label, "model": model, **r, "scored": flags})
        if (i + 1) % 25 == 0:
            print(f"  {label}: {i + 1}/{n}", file=sys.stderr)
    return responses


def live_run(
    provider: str,
    n: int,
    evidence: dict[str, Any],
    allowed: list[str],
    model: str | None = None,
    arms: tuple[str, ...] = ("layer1", "layer2", "layer3", "layer3_stress"),
    scorer_ctx: tuple[set[str], dict[str, float], str | None] | None = None,
    log_dir: str | None = None,
    task_label: str = "synthetic",
    seed: int = 0,
    resume: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    kind, client, default_model = _build_live_client(provider)
    use_model = model or default_model
    print(f"  provider={provider} kind={kind} model={use_model}", file=sys.stderr)
    ctx = scorer_ctx or (set(allowed), evidence_correlation_map(evidence), top_candidate(evidence))
    ev_hash = _evidence_hash(evidence)
    out: dict[str, list[dict[str, Any]]] = {}
    for layer in arms:
        log = (
            RunLog(
                log_dir,
                provider,
                use_model,
                task_label,
                layer,
                n,
                seed,
                resume,
                evidence_hash=ev_hash,
            )
            if log_dir
            else None
        )
        out[layer] = _run_cell(
            use_model,
            n,
            ctx,
            log,
            layer,
            lambda layer=layer: _live_one_response(  # type: ignore[misc]
                kind, client, use_model, layer, evidence, allowed
            ),
        )
    return out


def sweep_run(
    provider: str,
    n: int,
    evidence: dict[str, Any],
    allowed: list[str],
    model: str | None = None,
    scorer_ctx: tuple[set[str], dict[str, float], str | None] | None = None,
    log_dir: str | None = None,
    task_label: str = "synthetic",
    seed: int = 0,
    resume: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Run the bare-prompt paraphrase sweep: same open tool, same evidence,
    same model — only the (rule-free) system prompt varies."""
    kind, client, default_model = _build_live_client(provider)
    use_model = model or default_model
    print(
        f"  sweep: provider={provider} model={use_model}, {len(SWEEP_BARE_VARIANTS)} variants",
        file=sys.stderr,
    )
    ctx = scorer_ctx or (set(allowed), evidence_correlation_map(evidence), top_candidate(evidence))

    def _one_sweep(prompt: str) -> dict[str, Any]:
        try:
            if kind == "openai":
                response = client.chat.completions.create(
                    model=use_model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": _user_message(evidence)},
                    ],
                    tools=[_open_submit_tool_openai()],
                    tool_choice="required",
                )
                return _input_from_openai(response)
            response = client.messages.create(
                model=use_model,
                max_tokens=1024,
                system=prompt,
                tools=[_open_submit_tool_anthropic()],
                tool_choice={"type": "tool", "name": SUBMIT_TOOL_NAME},
                messages=[{"role": "user", "content": _user_message(evidence)}],
            )
            return _input_from_anthropic(response)
        except Exception:  # noqa: BLE001
            return _normalize({})

    ev_hash = _evidence_hash(evidence)
    out: dict[str, list[dict[str, Any]]] = {}
    for name, prompt in SWEEP_BARE_VARIANTS:
        log = (
            RunLog(
                log_dir,
                provider,
                use_model,
                task_label,
                f"sweep-{name}",
                n,
                seed,
                resume,
                evidence_hash=ev_hash,
            )
            if log_dir
            else None
        )
        out[name] = _run_cell(
            use_model,
            n,
            ctx,
            log,
            name,
            lambda prompt=prompt: _one_sweep(prompt),  # type: ignore[misc]
        )
    return out


# --------------------------------------------------------------------------- #
# Scoring                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class MetricResult:
    metric: str
    n: int
    k: int
    rate: float
    ci_low: float
    ci_high: float


@dataclass
class LayerResult:
    layer: str
    metrics: dict[str, MetricResult]


def _metric(metric: str, k: int, n: int) -> MetricResult:
    low, high = wilson_ci_95(k, n)
    return MetricResult(
        metric=metric, n=n, k=k, rate=(k / n if n else 0.0), ci_low=low, ci_high=high
    )


def score_one(
    response: dict[str, Any],
    allowed_set: set[str],
    corr_map: dict[str, float],
    anchor: str | None,
) -> dict[str, bool]:
    """Score a single response on the three contract channels.

    entity — references >= 1 column (in columns_referenced or a claim)
        outside the evidence set.
    value  — carries >= 1 claim whose column IS in the evidence but whose
        value differs from the measured one by > VALUE_TOLERANCE.
    omission — commits to a verdict other than cannot_determine yet never
        references the top-ranked candidate column. For contract arms the
        contract's own flag ("omitted") is authoritative.
    """
    cols = list(response.get("columns") or [])
    claims = [c for c in (response.get("claims") or []) if isinstance(c, dict)]
    claim_cols = [str(c.get("column", "")) for c in claims]
    entity = any(c not in allowed_set for c in cols + claim_cols)
    value = False
    for c in claims:
        col = str(c.get("column", ""))
        val = c.get("value")
        if col in corr_map and (
            not isinstance(val, (int, float)) or abs(float(val) - corr_map[col]) > VALUE_TOLERANCE
        ):
            value = True
            break
    if response.get("omitted") is not None:
        omission = bool(response["omitted"])
    else:
        committed = bool(cols or claims) and response.get("verdict") not in (
            "",
            "cannot_determine",
        )
        omission = committed and anchor is not None and anchor not in set(cols) | set(claim_cols)
    return {"entity": entity, "value": value, "omission": omission}


def score_responses(
    layer: str,
    responses: list[dict[str, Any]],
    allowed_set: set[str],
    corr_map: dict[str, float],
    anchor: str | None,
) -> LayerResult:
    """Aggregate :func:`score_one` over a layer's responses (Wilson CIs)."""
    n = len(responses)
    k_entity = k_value = k_omit = 0
    for r in responses:
        flags = score_one(r, allowed_set, corr_map, anchor)
        k_entity += 1 if flags["entity"] else 0
        k_value += 1 if flags["value"] else 0
        k_omit += 1 if flags["omission"] else 0
    return LayerResult(
        layer=layer,
        metrics={
            "entity": _metric("entity", k_entity, n),
            "value": _metric("value", k_value, n),
            "omission": _metric("omission", k_omit, n),
        },
    )


LAYER_LABELS = {
    "layer1": "L1 bare prompt",
    "layer2": "L1+2 strict prompt",
    "layer3": "L1+2+3 evidence-bound",
    "layer3_stress": "STRESS bare+TierB only",
    "tier_a": "TIER-A only (enum, no verify)",
    "stress_mech": "L3 + worst paraphrase",
}

METRIC_LABELS = {"entity": "Entity-fab", "value": "Value-fab", "omission": "Omission"}


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reproduce the three-layer phantom-column fabrication ablation."
    )
    ap.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="live = real API (produces Table I). mock = illustrative no-API demo.",
    )
    ap.add_argument(
        "--n", type=int, default=200, help="Samples per contract layer (paper uses 200)."
    )
    ap.add_argument("--seed", type=int, default=0, help="Seed for the synthetic frame / mock RNG.")
    ap.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        default="anthropic",
        help=(
            "Live provider. All except 'anthropic' use the OpenAI-compatible "
            "protocol via their base_url; 'vllm' targets a local server."
        ),
    )
    ap.add_argument("--model", default=None, help="Override the provider's default model.")
    ap.add_argument(
        "--check-providers",
        action="store_true",
        help="Print key/env/model wiring for every provider and exit (zero API calls).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run plan and its estimated cost, then exit (zero API calls).",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Wiring validation: N=5, arms layer1+layer3 only (~10 cheap calls).",
    )
    ap.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="Abort before any live call if the estimated cost exceeds this budget.",
    )
    ap.add_argument(
        "--log-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"),
        help="Directory for incremental JSONL run logs (paid responses survive crashes).",
    )
    ap.add_argument("--no-log", action="store_true", help="Disable the JSONL run log.")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted live run from its JSONL log instead of restarting.",
    )
    ap.add_argument(
        "--task",
        choices=["synthetic", "csv", "case"],
        default="synthetic",
        help=(
            "Evidence source: the synthetic frame, a real CSV with an injected "
            "leak (--csv-path/--target/--injector), or a real-world case study "
            "with a NATURAL documented leak (--case; analysis_plan A2)."
        ),
    )
    ap.add_argument(
        "--case",
        default=None,
        help="Case study for --task case: 'bodyfat' (real leak) or 'sambanis' (negative control).",
    )
    ap.add_argument("--csv-path", default=None, help="Real dataset CSV for --task csv.")
    ap.add_argument("--target", default=None, help="Numeric target column for --task csv.")
    ap.add_argument(
        "--injector",
        default="monotone_log",
        choices=list_injectors(),
        help=(
            "Leak pattern planted into the CSV task (FabBench, Phase 2). "
            "Default reproduces the June 2026 log-of-target task."
        ),
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="Run the 6-variant bare-prompt paraphrase sweep (live only).",
    )
    ap.add_argument(
        "--no-stress",
        action="store_true",
        help="Skip the layer3_stress arm (bare prompt + Tier B only).",
    )
    ap.add_argument(
        "--only-extra",
        action="store_true",
        help="Run ONLY the diagnostic arms tier_a and stress_mech (live only).",
    )
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = ap.parse_args()

    # ---------------- Zero-cost wiring check ------------------------------- #
    if args.check_providers:
        print("| provider  | key env             | set | model                        | base_url |")
        print("| --------- | ------------------- | --- | ---------------------------- | -------- |")
        for name in sorted(PROVIDERS):
            cfg = PROVIDERS[name]
            has_key = (
                "yes"
                if os.environ.get(cfg["key_env"])
                else ("opt" if cfg.get("key_optional") else "NO")
            )
            print(
                f"| {name:<9s} | {cfg['key_env']:<19s} | {has_key:<3s} "
                f"| {cfg['model']:<28s} | {cfg['base_url'] or '(default)'} |"
            )
        print("\nNo API calls were made. 'NO' = export the key before a live run.")
        return 0

    if (args.dry_run or args.max_cost_usd is not None) and args.mode != "live":
        raise SystemExit(
            "--dry-run/--max-cost-usd concern live spending; add --mode live "
            "(mock mode is already free)."
        )
    if args.smoke:
        if args.resume:
            raise SystemExit(
                "--smoke validates CURRENT wiring with fresh calls; --resume "
                "would replay a stale log and prove nothing. Drop --resume."
            )
        args.n = 5
        args.no_log = True  # repeatable; 5-call smoke logs have no audit value

    use_model = args.model or PROVIDERS[args.provider]["model"]

    if args.task == "csv":
        if not args.csv_path or not args.target:
            raise SystemExit("--task csv requires --csv-path and --target.")
        evidence = build_csv_evidence(
            args.csv_path, args.target, seed=args.seed, injector=args.injector
        )
        task_label = f"csv:{os.path.basename(args.csv_path)}/{args.target}/{args.injector}"
    elif args.task == "case":
        if not args.case:
            raise SystemExit("--task case requires --case (bodyfat or sambanis).")
        if args.injector != "monotone_log":
            raise SystemExit("--injector does not apply to case studies (natural leaks).")
        from fetch_fabbench_datasets import build_case_evidence

        evidence = build_case_evidence(args.case)
        task_label = f"case:{args.case}"
    else:
        if args.injector != "monotone_log":
            raise SystemExit(
                "--injector applies to --task csv; the synthetic task is the "
                "frozen June 2026 configuration."
            )
        evidence = build_synthetic_evidence(seed=args.seed)
        task_label = "synthetic"
    allowed = evidence_allowed_columns(evidence)
    allowed_set = set(allowed)
    corr_map = evidence_correlation_map(evidence)
    anchor = top_candidate(evidence)

    print(
        f"Running {args.mode} {'sweep' if args.sweep else 'ablation'}, task={task_label}, "
        f"N={args.n} (evidence columns: {len(allowed)}, anchor: {anchor})...",
        file=sys.stderr,
    )

    log_dir = None if (args.no_log or args.mode != "live") else args.log_dir
    scorer_ctx = (allowed_set, corr_map, anchor)

    def _cost_banner(cells: tuple[str, ...], per_cell_n: int) -> None:
        """Print the cost estimate; enforce --max-cost-usd; exit on --dry-run."""
        est, calls = estimate_cost_usd(use_model, cells, per_cell_n)
        est_str = f"~${est:.2f}" if est is not None else "UNKNOWN (model not in PRICING_PER_M)"
        print(
            f"  plan: {len(cells)} cell(s) x N={per_cell_n} -> ~{calls} calls, "
            f"ESTIMATED cost {est_str} (model={use_model}; rough token estimate, "
            "not a bill)",
            file=sys.stderr,
        )
        if args.dry_run:
            print("  --dry-run: no API calls were made.", file=sys.stderr)
            raise SystemExit(0)
        if args.max_cost_usd is not None:
            if est is None:
                raise SystemExit(
                    "--max-cost-usd set but pricing for this model is unknown; "
                    "add it to PRICING_PER_M or drop the flag."
                )
            if est > args.max_cost_usd:
                raise SystemExit(
                    f"Estimated cost ${est:.2f} exceeds --max-cost-usd "
                    f"{args.max_cost_usd:.2f}; aborting before any live call."
                )

    # ---------------- Sweep mode: bare-prompt paraphrase distribution -------- #
    if args.sweep:
        if args.mode != "live":
            raise SystemExit("--sweep is a live measurement; add --mode live.")
        if args.smoke:
            raise SystemExit("--smoke runs the ablation arms; drop --sweep.")
        _cost_banner(tuple(name for name, _ in SWEEP_BARE_VARIANTS), args.n)
        sweep = sweep_run(
            args.provider,
            args.n,
            evidence,
            allowed,
            model=args.model,
            scorer_ctx=scorer_ctx,
            log_dir=log_dir,
            task_label=task_label,
            seed=args.seed,
            resume=args.resume,
        )
        rows = [
            (name, score_responses(name, sweep[name], allowed_set, corr_map, anchor))
            for name, _ in SWEEP_BARE_VARIANTS
        ]
        print(f"\n## Bare-prompt paraphrase sweep ({task_label}, N = {args.n} per variant)\n")
        print("| Variant     | Entity-fab | Wilson 95% CI    | k / N      |")
        print("| ----------- | :--------: | :--------------: | :--------: |")
        rates = []
        for name, r in rows:
            m = r.metrics["entity"]
            rates.append(m.rate)
            print(
                f"| {name:<11s} | {m.rate * 100:>8.1f}% "
                f"| [{m.ci_low * 100:>5.2f}, {m.ci_high * 100:>5.2f}] | {m.k:>3d} / {m.n:<4d} |"
            )
        rates.sort()
        print(
            f"\nSpread across {len(rates)} rule-free paraphrases of the same task: "
            f"min {rates[0] * 100:.1f}%, max {rates[-1] * 100:.1f}%. "
            "Same model, same evidence, same tool schema — "
            "only the wording varies."
        )
        return 0

    # ---------------- Layer ablation --------------------------------------- #
    arms: tuple[str, ...] = ("layer1", "layer2", "layer3", "layer3_stress")
    if args.no_stress or args.mode == "mock":
        arms = ("layer1", "layer2", "layer3")
    if args.only_extra:
        if args.mode != "live":
            raise SystemExit("--only-extra is a live measurement; add --mode live.")
        arms = ("tier_a", "stress_mech")
    if args.smoke:
        if args.mode != "live":
            raise SystemExit("--smoke validates live wiring; add --mode live.")
        arms = ("layer1", "layer3")

    if args.mode == "mock":
        responses_by_layer = mock_run(args.n, allowed, corr_map, anchor, seed=args.seed)
    else:
        _cost_banner(arms, args.n)
        responses_by_layer = live_run(
            args.provider,
            args.n,
            evidence,
            allowed,
            model=args.model,
            arms=arms,
            scorer_ctx=scorer_ctx,
            log_dir=log_dir,
            task_label=task_label,
            seed=args.seed,
            resume=args.resume,
        )

    results = [
        score_responses(layer, responses_by_layer[layer], allowed_set, corr_map, anchor)
        for layer in arms
    ]

    # Tier B catch telemetry — how often the contract demonstrably fired.
    rejection_summary: dict[str, dict[str, int]] = {}
    for layer in arms:
        rej = [int(r.get("rejections") or 0) for r in responses_by_layer[layer]]
        rejection_summary[layer] = {
            "responses_with_catches": sum(1 for x in rej if x > 0),
            "total_catches": sum(rej),
        }

    if args.smoke:
        # Wiring summary: did calls succeed, how slow, how many tokens, what
        # would a real battery cost at these observed sizes?
        print("\n  smoke wiring summary (NOT a measurement):", file=sys.stderr)
        for layer in arms:
            rs = responses_by_layer[layer]
            lat = [r["latency_ms"] for r in rs if r.get("latency_ms") is not None]
            uin = [r["usage_in"] for r in rs if r.get("usage_in") is not None]
            uout = [r["usage_out"] for r in rs if r.get("usage_out") is not None]
            empty = sum(1 for r in rs if not r.get("columns") and not r.get("claims"))
            print(
                f"    {layer}: {len(rs)} calls, {empty} empty responses, "
                f"avg latency {sum(lat) / len(lat):.0f} ms"
                if lat
                else f"    {layer}: {len(rs)} calls, {empty} empty responses",
                file=sys.stderr,
            )
            if uin and uout:
                pricing = PRICING_PER_M.get(use_model)
                extra = ""
                if pricing:
                    per_call = (
                        sum(uin) / len(uin) * pricing[0] + sum(uout) / len(uout) * pricing[1]
                    ) / 1_000_000.0
                    extra = f", ~${per_call * 1000:.3f}/1k calls"
                print(
                    f"      avg tokens in/out: {sum(uin) / len(uin):.0f}/"
                    f"{sum(uout) / len(uout):.0f}{extra}",
                    file=sys.stderr,
                )
        print(
            "  smoke OK means: key valid, tool-call path works, responses parse. "
            "If 'empty responses' is high, inspect the run log before spending "
            "on a full battery.\n",
            file=sys.stderr,
        )

    if args.json:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "provider": args.provider if args.mode == "live" else None,
                    "model": use_model if args.mode == "live" else None,
                    "task": task_label,
                    "n_per_layer": args.n,
                    "seed": args.seed,
                    "evidence_columns": allowed,
                    "anchor": anchor,
                    "value_tolerance": VALUE_TOLERANCE,
                    "tier_b_catches": rejection_summary,
                    "results": [
                        {
                            "layer": r.layer,
                            "label": LAYER_LABELS[r.layer],
                            "metrics": {
                                name: {
                                    "n": m.n,
                                    "k": m.k,
                                    "rate": m.rate,
                                    "ci_low": m.ci_low,
                                    "ci_high": m.ci_high,
                                }
                                for name, m in r.metrics.items()
                            },
                        }
                        for r in results
                    ],
                },
                indent=2,
            )
        )
        return 0

    if args.mode == "mock":
        print("\n[!] MOCK MODE — illustrative only. These are NOT measured results;")
        print("    value/omission rates are placeholders and Layer 3's zeros are by")
        print("    construction of the simulator. Use --mode live for Table I.\n")

    print(f"## Three-channel violation rates ({args.mode}, task={task_label}, N = {args.n})\n")
    for metric in ("entity", "value", "omission"):
        print(f"### {METRIC_LABELS[metric]} rate")
        print("| Layer                   | Rate  | Wilson 95% CI    | k / N      |")
        print("| ----------------------- | :---: | :--------------: | :--------: |")
        for r in results:
            m = r.metrics[metric]
            print(
                f"| {LAYER_LABELS[r.layer]:<23s} | {m.rate * 100:>4.1f}% "
                f"| [{m.ci_low * 100:>5.2f}, {m.ci_high * 100:>5.2f}] | {m.k:>3d} / {m.n:<4d} |"
            )
        print()
    print("### Tier B catches (deterministic rejections that fired)")
    print("| Layer                   | Responses with >=1 catch | Total catches |")
    print("| ----------------------- | :----------------------: | :-----------: |")
    for layer in arms:
        s = rejection_summary[layer]
        print(
            f"| {LAYER_LABELS[layer]:<23s} | {s['responses_with_catches']:>9d} / {args.n:<6d} "
            f"| {s['total_catches']:>9d}     |"
        )
    print(
        "\nLayer 3 routes through the shipped investigate_leakage_bound: column enums "
        "are bound to the evidence at call time and Tier B deterministically verifies "
        "entity soundness, claim values (tolerance "
        f"{VALUE_TOLERANCE}), and completeness; persistent omissions are flagged. "
        "The STRESS arm runs the same shipped contract with the BARE prompt and "
        "WITHOUT Tier A enums — its user-facing rates plus its catch counts show "
        "Tier B doing the work alone. See paper Section III."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
