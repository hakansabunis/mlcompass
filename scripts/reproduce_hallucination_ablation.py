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

Baseline comparison arms (``--only-baselines``; analysis_plan.md §6). The
ablation above shows what each LAYER of the contract buys; the baselines show
what the contract buys over the alternatives a reviewer will name. All of them
run the same task, the same evidence, the same model and the same three-channel
scoring, and differ only in the enforcement mechanism:

  * ``layer1`` — the no-enforcement control. Already exactly that (bare prompt,
    open enum-free schema, no validation, no retry), and the baselines are
    built on that same call, so it is reused rather than duplicated.
  * ``layer3_bare`` — the CONTRACT arm of this comparison (plan 2026-09 §2.1
    ``A-CONTRACT``): the shipped enforcement stack (Tier A call-time enum +
    Tier B verification) with the faithfulness PROMPT removed, so the contract
    is compared against bare-prompt baselines without confounding prompt with
    mechanism. Shipped L3 carries the strict prompt and is NOT the comparator
    for this block.
  * ``guardrails_stock`` — Guardrails AI with structural validation and reask
    only: what a validate-and-reask toolkit gives you out of the box.
  * ``guardrails_tierb`` — Guardrails AI running custom validators that encode
    the Tier B checks. Our validator, their loop; the comparison is loops.
  * ``static_schema_noenum`` — the registered ``A-STRICT-STATIC`` arm: the
    provider's strict structured-output mode over a static schema carrying
    types and required fields and NO column enum at all.
  * ``static_schema`` — a SECOND, separately registered arm
    (``A-STATIC-ENUM-STALE``, plan 2026-09 §9 amendment A1): the same strict
    mode, but with a column enum declared once at author time instead of
    generated from the evidence at call time. It isolates a STALE domain
    against a live one, which the no-enum arm cannot see. The two test
    different hypotheses and are never reported under one label.

See ``scripts/baseline_guardrails.py`` for the toolkit configuration and the
reasons behind each choice.

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

Run logs are the evidence, not a by-product (plan 2026-09 §6, §8.4). Two facts
about them, recorded here because they are easy to get wrong:

  * ``scripts/runs/`` is TRACKED. The repository ignores ``runs/`` globally
    (ML artefacts) and re-includes this one with a ``!scripts/runs/``
    negation, which works because the negation names the directory git would
    otherwise refuse to descend into. Verified, not assumed: ``git
    check-ignore`` reports a fresh log there as NOT ignored.
  * **No JSONL from the June 2026 battery survives on disk.** The records
    behind paper Tables I-II are gone, so the manuscript's "raw live-run
    records are in the repository" is not currently true, and no pre-v2 cell
    exists to be compared against (see RECORD_SCHEMA_VERSION). The Round-2
    battery is what rebuilds that claim; until it runs and its logs are
    committed, the sentence should not ship.

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

    # zero-cost: what would the baseline comparison battery cost?
    python scripts/reproduce_hallucination_ablation.py --mode live \
        --only-baselines --n 200 --dry-run

    # pennies: validate every baseline arm end to end
    python scripts/reproduce_hallucination_ablation.py --mode live \
        --only-baselines --smoke

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

from baseline_guardrails import GUARDRAILS_ARMS  # noqa: E402
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


def build_synthetic_crowded_evidence(seed: int = 0) -> dict[str, Any]:
    """Frozen instance #14 — the value-channel stressor (analysis_plan A3.4).

    Ten features whose measured correlations sit on an EXACT 0.003-spaced grid
    in the 0.900-0.924 band plus one 0.995 anchor (``crowded_frame`` in
    ``fabbench_injectors`` — the construction is Gram-Schmidt-exact, not
    sampled luck). Restating any specific value from prose is error-prone
    here; the entity channel is unaffected. The suspicious metric is COMPUTED
    from the frame's own predictions (A3.5 discipline).
    """
    import numpy as np
    from fabbench_injectors import crowded_frame

    from mlcompass.tools.leakage import detect_leakage

    df, target = crowded_frame(seed=seed)
    y = df[target].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot

    return detect_leakage(
        df,
        y_true_col=target,
        y_pred_col="y_pred",
        task="regression",
        suspicious_metric={"name": "r2", "value": round(r2, 4)},
    )


def build_csv_evidence(
    csv_path: str,
    target: str,
    seed: int = 0,
    injector: str = "monotone_log",
    pseudonymize: bool = False,
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

    if pseudonymize:
        # A3.10 dataset-familiarity arm: hash EVERY column name (target and
        # injected anchor included) so the narrator cannot lean on memorized
        # public-dataset schemas; only the numbers remain informative. The
        # mapping is deterministic (content hash of the name) and printed for
        # the audit trail.
        mapping = {
            c: f"col_{hashlib.sha256(str(c).encode('utf-8')).hexdigest()[:8]}"
            for c in df.columns
            if c != "y_pred"
        }
        df = df.rename(columns=mapping)
        target = mapping[target]
        print(f"  pseudonymized columns: {json.dumps(mapping)}", file=sys.stderr)

    # A3.5: the suspicious metric is COMPUTED from these predictions, never
    # asserted, so the narrator never sees a number nobody measured.
    y_pred_arr = df["y_pred"].to_numpy(dtype=float)
    ss_res = float(np.sum((y_arr - y_pred_arr) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot

    return detect_leakage(
        df,
        y_true_col=target,
        y_pred_col="y_pred",
        task="regression",
        suspicious_metric={"name": "r2", "value": round(r2, 4)},
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
        # Back to 'deepseek-chat'. The 2026-07-24 repin to a direct name was
        # made on a deprecation notice and never exercised: every current
        # direct name is a thinking-mode model, and thinking mode rejects the
        # forced tool_choice this harness requires --
        #     400 "Thinking mode does not support this tool_choice"
        # -- so '--provider deepseek' produced 100% transport errors and
        # scored every cell 0/0. Measured 2026-09-15 against deepseek-chat,
        # deepseek-flash, deepseek-v4-flash and deepseek-v4-pro: only
        # deepseek-chat accepts the call. It is absent from models.list() but
        # still served, which is why a listing check would have missed this.
        # Re-pin only after probing the replacement with a forced tool_choice.
        "model": "deepseek-chat",
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
#
# Baseline arms (analysis_plan §6): guardrails_stock reasks only on a
# STRUCTURAL failure, which a forced tool call makes rare (~1.05).
# guardrails_tierb reasks on any faithfulness violation under the bare prompt
# and additionally requires the anchor unconditionally, so it is budgeted at
# the STRESS-like 1.5 — deliberately an OVER-estimate, because an
# under-estimated baseline is the one that blows a budget. The static arms are
# a single call with no retry loop.
#
# layer3_bare carries Tier A, so entity violations are suppressed in the schema
# and cannot drive retries the way they do in layer3_stress; its retries come
# from the value and omission checks under a BARE prompt, which layer3's strict
# prompt suppresses. It is budgeted at the stress factor rather than layer3's
# 1.05 on the same principle as guardrails_tierb: an over-estimated arm costs a
# cautious banner, an under-estimated one blows a budget mid-run. Replace it
# with the observed calls/response from --smoke before committing to a battery.
ARM_CALL_FACTOR: dict[str, float] = {
    "layer1": 1.0,
    "layer2": 1.0,
    "layer3": 1.05,
    "layer3_bare": 1.45,
    "layer3_stress": 1.45,
    "layer3_stress_generic": 1.45,
    "tier_a": 1.0,
    "stress_mech": 1.05,
    "guardrails_stock": 1.05,
    "guardrails_tierb": 1.5,
    "static_schema": 1.0,
    "static_schema_noenum": 1.0,
}

# Guardrails prepends its JSON-schema scaffolding to every REASK prompt, so a
# reask costs meaningfully more input tokens than a first attempt. The
# --dry-run estimator is deliberately rough (see EST_IN_TOKENS_PER_CALL); this
# multiplier keeps the Guardrails estimate from reading too low. Replace the
# estimate with the observed per-call token counts from --smoke before
# committing to a battery.
ARM_INPUT_TOKEN_FACTOR: dict[str, float] = {
    "guardrails_stock": 1.6,
    "guardrails_tierb": 1.6,
}

# --------------------------------------------------------------------------- #
# Static (author-time) schemas — the two provider-strict baselines             #
# --------------------------------------------------------------------------- #

# Two DIFFERENT arms live here and they are never merged, because they test
# different hypotheses (plan 2026-09 §2.1, and §9 amendment A1):
#
#   static_schema_noenum = A-STRICT-STATIC, the REGISTERED arm. Provider strict
#       mode over a schema carrying types and required fields and NO column
#       enum whatsoever, so the column domain is unconstrained and any invented
#       name decodes successfully. Tests H8, "a static strict schema constrains
#       shape, not content", whose rejecting outcome is 0/200 on entity while
#       the matched A-L1 cell exceeds 3%.
#
#   static_schema       = A-STATIC-ENUM-STALE, an ADDITIONAL arm. Same strict
#       mode, but with a column enum frozen at author time. Tests something H8
#       cannot see: what a STALE domain does once the task moves on. This is
#       the deployment-realistic shape of "declare the schema once" and the arm
#       closest to the manuscript's call-time-binding claim.
#
# Neither substitutes for the other and neither is ever reported under the
# other's arm id. Together with `tier_a --strict` they complete a three-rung
# ladder — no enum / stale enum / live enum — that is a one-variable contrast
# at every step, because all three run the SAME product tool builder.
#
# The `static_schema` arm answers "what does binding the enum to the evidence
# AT CALL TIME buy over declaring the schema once?". It is the SAME product
# code path as the `tier_a` arm (build_submit_investigation_tool*, strict
# variant) with one difference: the enum domain is this frozen list instead of
# `evidence_allowed_columns(evidence)`.
#
# The list is the evidence column set of the FROZEN REFERENCE TASK (the June
# 2026 synthetic monotone_log instance, seed 0) — i.e. exactly what a careful
# engineer would hard-code after looking at the dataset the application was
# authored against. Choosing it this way is what keeps the comparison fair:
#
#   * On the reference task the static and dynamic domains are IDENTICAL, so
#     the arm is degenerate there by construction (any difference would be
#     noise, not mechanism). The harness says so out loud at run time.
#   * On every other task instance the static domain is stale, which is the
#     real deployment condition the dynamic binding exists to handle.
#
# Picking an arbitrary or deliberately wrong list instead would manufacture a
# 100% fabrication rate and would be a rigged comparison. Every cell records
# its own coverage numbers (see `_static_schema_coverage`) so a reader can
# check the degree of mismatch that produced the rate, per cell.
STATIC_SCHEMA_COLUMNS: tuple[str, ...] = (
    "feature_10",
    "feature_12",
    "feature_4",
    "feature_5",
    "feature_6",
    "feature_7",
    "feature_8",
    "feature_9",
    "log_target_v2",
    "near_target_proxy",
)


# The two static arms. Only the first declares a column domain at all.
STATIC_SCHEMA_ARMS: tuple[str, ...] = ("static_schema", "static_schema_noenum")


def _static_schema_coverage(
    static_columns: list[str], evidence_columns: list[str], anchor: str | None
) -> dict[str, Any]:
    """Per-cell diagnostic for how well the frozen domain fits this evidence.

    Recorded on every `static_schema` response so a rate can never be read
    without the domain mismatch that produced it.
    """
    static_set, evidence_set = set(static_columns), set(evidence_columns)
    overlap = sorted(static_set & evidence_set)
    return {
        "n_static": len(static_set),
        "n_evidence": len(evidence_set),
        "overlap": len(overlap),
        "coverage": (len(overlap) / len(evidence_set)) if evidence_set else 0.0,
        "anchor_in_static": (anchor in static_set) if anchor is not None else None,
        "evidence_not_in_static": sorted(evidence_set - static_set),
        "static_not_in_evidence": sorted(static_set - evidence_set),
    }


def estimate_cost_usd(model: str, arms: tuple[str, ...], n: int) -> tuple[float | None, int]:
    """Return (estimated USD or None if pricing unknown, estimated call count).

    Arms that inflate the prompt on retry (the Guardrails baselines, which
    prepend their JSON-schema scaffolding to every reask) are costed with
    ARM_INPUT_TOKEN_FACTOR so the estimate does not read low.
    """
    per_arm_calls = {arm: int(math.ceil(n * ARM_CALL_FACTOR.get(arm, 1.0))) for arm in arms}
    calls = sum(per_arm_calls.values())
    pricing = PRICING_PER_M.get(model)
    if pricing is None:
        return None, calls
    p_in, p_out = pricing
    usd = 0.0
    for arm, arm_calls in per_arm_calls.items():
        tokens_in = EST_IN_TOKENS_PER_CALL * ARM_INPUT_TOKEN_FACTOR.get(arm, 1.0)
        usd += arm_calls * (tokens_in * p_in + EST_OUT_TOKENS_PER_CALL * p_out) / 1_000_000.0
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
            "provider_calls": int(result.get("attempts_made") or 1),
        }
    )
    return out


def _error_response(exc: Exception) -> dict[str, Any]:
    """A transport-failure marker (A3.9b): NOT data. Scoring excludes these
    (they would otherwise count as clean non-fabricating responses,
    contradicting preregistration §7); they are logged with their reason and
    reported per cell."""
    r = _normalize({})
    r["error"] = f"{type(exc).__name__}: {exc}"
    return r


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
        # Provider calls this response cost. 1 for every single-shot arm; the
        # contract path overrides it with attempts_made and the Guardrails
        # arms with the toolkit's own call count, so "calls per response" is
        # comparable across mechanisms (analysis_plan §6 requires calls/cost
        # alongside the three channels). ADDITIVE record field: cells logged
        # before it existed simply lack it and stay comparable, because no
        # channel definition changed.
        "provider_calls": 1,
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


def _split_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Split an OpenAI-style message list into (system text, chat turns).

    Guardrails hands the transport one flat message list; the Anthropic
    Messages API takes the system prompt as its own parameter. Multiple system
    turns (Guardrails adds its own on a reask) are concatenated in order.
    """
    system = "\n\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    turns = [
        {"role": str(m.get("role")), "content": str(m.get("content", ""))}
        for m in messages
        if m.get("role") != "system"
    ]
    return system, turns or [{"role": "user", "content": ""}]


def _raw_tool_input(kind: str, response: Any) -> tuple[dict[str, Any], int | None, int | None]:
    """The tool call's RAW argument dict plus token usage.

    The baseline arms hand the model's payload to an external toolkit, which
    parses it against its own schema, so the payload must reach it verbatim —
    :func:`_normalize` would drop ``narration``/``confidence`` and the toolkit
    would then reask about fields the model actually supplied, manufacturing
    reasks that never happened.
    """
    if kind == "openai":
        usage = getattr(response, "usage", None)
        tokens = (
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )
        calls = getattr(response.choices[0].message, "tool_calls", None) or []
        if not calls:
            return {}, tokens[0], tokens[1]
        args = calls[0].function.arguments
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
        except (json.JSONDecodeError, TypeError):
            return {}, tokens[0], tokens[1]
        return (dict(parsed) if isinstance(parsed, dict) else {}), tokens[0], tokens[1]

    usage = getattr(response, "usage", None)
    tokens = (getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None))
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            raw = getattr(block, "input", {})
            return (dict(raw) if isinstance(raw, dict) else {}), tokens[0], tokens[1]
    return {}, tokens[0], tokens[1]


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
    kind: str,
    client: Any,
    model: str,
    layer: str,
    evidence: dict[str, Any],
    allowed: list[str],
    strict: bool = False,
    temperature: float | None = 1.0,
    guardrails_reasks: int = 2,
    static_columns: tuple[str, ...] = STATIC_SCHEMA_COLUMNS,
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

    ``temperature`` is the sampling pin (default 1.0): sent verbatim where the
    provider accepts it; on a parameter rejection (reasoning endpoints) the pin
    is dropped for the retry and the record's ``sampling`` field says so. Every
    returned record carries ``sampling.temperature`` = what was actually sent.
    """
    temp = temperature

    def _sampled(r: dict[str, Any]) -> dict[str, Any]:
        r["sampling"] = {"temperature": temp}
        return r

    def _param_rejected(e: Exception) -> bool:
        # e.g. "Unsupported parameter: 'temperature' is not supported with
        # this model." — deterministic, so retrying WITH the pin cannot work.
        return temp is not None and "temperature" in str(e).lower()

    def _strict_rejected(e: Exception) -> bool:
        # A provider that does not implement strict structured outputs rejects
        # the flag (or the strict-compatible schema) deterministically. The
        # static_schema arm degrades to non-strict and SAYS SO in the record
        # rather than dropping the cell.
        text = str(e).lower()
        return "strict" in text or "additionalproperties" in text

    if layer in GUARDRAILS_ARMS:
        # BASELINE (analysis_plan §6) — validate-and-reask toolkit. Same bare
        # prompt, same evidence, same OPEN tool schema, same sampling pin as
        # L1: the ONLY difference is that the toolkit's loop sits around the
        # call. `guardrails_stock` validates structure only (what the toolkit
        # gives you out of the box); `guardrails_tierb` adds validators that
        # encode the Tier B checks, so the comparison is loops, not checkers.
        import baseline_guardrails

        tool = _open_submit_tool_openai() if kind == "openai" else _open_submit_tool_anthropic()

        def _transport(
            messages: list[dict[str, Any]],
        ) -> tuple[dict[str, Any], int | None, int | None]:
            """One provider call on Guardrails' behalf. Raises on persistent
            transport failure so the arm records an error marker (A3.9b)
            rather than scoring a dropped call as a clean response."""
            nonlocal temp
            last: Exception | None = None
            for _attempt in range(2):
                try:
                    if kind == "openai":
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            tools=[tool],
                            tool_choice="required",
                            **({} if temp is None else {"temperature": temp}),
                        )
                    else:
                        system_text, turns = _split_system(messages)
                        response = client.messages.create(
                            model=model,
                            max_tokens=1024,
                            system=system_text,
                            tools=[tool],
                            tool_choice={"type": "tool", "name": SUBMIT_TOOL_NAME},
                            messages=turns,
                            **({} if temp is None else {"temperature": temp}),
                        )
                    return _raw_tool_input(kind, response)
                except Exception as e:  # noqa: BLE001 — SDK exception types vary
                    last = e
                    if _param_rejected(e):
                        temp = None
            raise last or RuntimeError("unknown transport failure")

        try:
            result = baseline_guardrails.run_guardrails_once(
                _transport,
                system_prompt=LIVE_SYSTEM_PROMPT_BARE,
                user_message=_user_message(evidence),
                allowed_columns=list(allowed),
                corr_map=evidence_correlation_map(evidence),
                anchor=top_candidate(evidence),
                tolerance=VALUE_TOLERANCE,
                with_validators=(layer == "guardrails_tierb"),
                num_reasks=guardrails_reasks,
            )
        except Exception as e:  # noqa: BLE001
            return _sampled(_error_response(e))
        record = _normalize({})
        record.update(result)
        return _sampled(record)

    if layer in STATIC_SCHEMA_ARMS:
        # BASELINE — provider strict mode over an AUTHOR-TIME schema. Same
        # product tool builder as `tier_a` and the same bare prompt; the only
        # difference is the column domain the schema declares:
        #
        #   static_schema_noenum (A-STRICT-STATIC): none at all — types and
        #       required fields only.
        #   static_schema (A-STATIC-ENUM-STALE): the frozen
        #       STATIC_SCHEMA_COLUMNS instead of this evidence's columns.
        #
        # That one-variable difference is what isolates the value of binding
        # the domain at call time. Both arms are strict BY CONSTRUCTION (strict
        # mode is the mechanism under test), independent of the global --strict
        # flag, which selects strict cells for the open arms.
        no_enum = layer == "static_schema_noenum"
        static_list = [] if no_enum else list(static_columns)

        def _static_tool(strict_on: bool, cols: list[str] = static_list) -> dict[str, Any]:
            builder = (
                build_submit_investigation_tool_openai
                if kind == "openai"
                else build_submit_investigation_tool
            )
            return builder(cols, enforce_enum=not no_enum, strict=strict_on)

        tool = _static_tool(True)
        coverage = (
            None
            if no_enum
            else _static_schema_coverage(static_list, list(allowed), top_candidate(evidence))
        )
        use_strict = True
        static_exc: Exception | None = None
        degraded: str | None = None
        for _attempt in range(3):
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
                        **({} if temp is None else {"temperature": temp}),
                    )
                    record = _input_from_openai(response)
                else:
                    response = client.messages.create(
                        model=model,
                        max_tokens=1024,
                        system=LIVE_SYSTEM_PROMPT_BARE,
                        tools=[tool],
                        tool_choice={"type": "tool", "name": SUBMIT_TOOL_NAME},
                        messages=[{"role": "user", "content": _user_message(evidence)}],
                        **({} if temp is None else {"temperature": temp}),
                    )
                    record = _input_from_anthropic(response)
                record["baseline"] = {
                    "mechanism": (
                        "provider-strict static schema, no call-time enum"
                        if no_enum
                        else "provider-strict static schema, author-time enum"
                    ),
                    "arm_id": ARM_IDS[layer],
                    "enum_declared": not no_enum,
                    "strict_requested": True,
                    "strict_applied": use_strict,
                    "degraded": degraded,
                    "static_columns": static_list,
                    "coverage": coverage,
                }
                return _sampled(record)
            except Exception as e:  # noqa: BLE001 — SDK exception types vary
                static_exc = e
                if _param_rejected(e):
                    temp = None
                elif use_strict and _strict_rejected(e):
                    # Report the degradation instead of silently dropping the
                    # response: the cell is no longer a strict-mode cell and
                    # every record says so.
                    use_strict = False
                    degraded = f"strict_unsupported: {type(e).__name__}: {e}"
                    print(
                        f"  {layer}: provider rejected strict tools; "
                        f"retrying WITHOUT strict and flagging the record "
                        f"({type(e).__name__}: {str(e)[:160]})",
                        file=sys.stderr,
                    )
                    tool = _static_tool(False)
        return _sampled(_error_response(static_exc or RuntimeError("unknown")))

    if layer == "tier_a":
        # Tier A in isolation: the evidence-bound enum is IN the schema, the
        # bare prompt states no rules, and Tier B verification is NOT applied.
        # Measures how much the runtime enum alone steers this endpoint.
        tool = (
            build_submit_investigation_tool_openai(allowed, strict=strict)
            if kind == "openai"
            else build_submit_investigation_tool(allowed, strict=strict)
        )
        last_exc: Exception | None = None
        for _attempt in range(2):
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
                        **({} if temp is None else {"temperature": temp}),
                    )
                    return _sampled(_input_from_openai(response))
                response = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=LIVE_SYSTEM_PROMPT_BARE,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": SUBMIT_TOOL_NAME},
                    messages=[{"role": "user", "content": _user_message(evidence)}],
                    **({} if temp is None else {"temperature": temp}),
                )
                return _sampled(_input_from_anthropic(response))
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if _param_rejected(e):
                    temp = None
        return _sampled(_error_response(last_exc or RuntimeError("unknown")))

    if layer == "stress_mech":
        # The FULL shipped contract (Tier A enums ON + Tier B) under the worst
        # naturally occurring paraphrase from the sweep (100% bare fabrication)
        # — answers "necessity shown only in an artificially weakened config".
        mech_prompt = dict(SWEEP_BARE_VARIANTS)["mechanical"]
        # Same transient-error policy as the open arms: a 429/timeout must
        # not kill a multi-cell battery mid-run (in-flight money + all
        # remaining cells); after two failures record an ERROR marker.
        last_exc = None
        for _attempt in range(2):
            try:
                result = investigate_leakage_bound(
                    evidence,
                    client=client,
                    model=model,
                    provider=("openai" if kind == "openai" else "anthropic"),
                    system_prompt=mech_prompt,
                    strict_tools=strict,
                    neutral_user=True,  # A3.11: sweep-comparable user message
                    temperature=temp,
                )
                return _sampled(_from_contract_result(result))
            except Exception as e:  # noqa: BLE001 — SDK exception types vary
                last_exc = e
                if _param_rejected(e):
                    temp = None
        return _sampled(_error_response(last_exc or RuntimeError("unknown")))

    if layer in ("layer3", "layer3_bare", "layer3_stress", "layer3_stress_generic"):
        stress = layer.startswith("layer3_stress")
        # `layer3_bare` is A-CONTRACT (plan 2026-09 §2.1): the SHIPPED
        # enforcement stack — Tier A call-time enums ON, Tier B verification
        # ON, the same repair budget — with the faithfulness prompt replaced
        # by the bare one and the contract-free user message. It is the
        # comparator in every registered contrast X1-X5, and it exists because
        # contrasting shipped L3 (strict prompt) against bare-prompt baselines
        # would confound prompt with mechanism. The ONLY difference from
        # `layer3` is the prompt pair; the only difference from `layer3_stress`
        # is that Tier A stays on.
        bare_prompt = stress or layer == "layer3_bare"
        last_exc = None
        for _attempt in range(2):
            try:
                result = investigate_leakage_bound(
                    evidence,
                    client=client,
                    model=model,
                    provider=("openai" if kind == "openai" else "anthropic"),
                    system_prompt=(LIVE_SYSTEM_PROMPT_BARE if bare_prompt else None),
                    enforce_schema_enum=not stress,
                    strict_tools=strict,
                    # A3.2: the generic-retry H5 control names nothing.
                    correction_style=("generic" if layer.endswith("_generic") else "named"),
                    # A3.11: bare-prompt arms use the bare (L1-aligned) user
                    # message, so propensity matches the baselines they are
                    # contrasted against.
                    neutral_user=bare_prompt,
                    temperature=temp,
                )
                return _sampled(_from_contract_result(result))
            except Exception as e:  # noqa: BLE001 — SDK exception types vary
                last_exc = e
                if _param_rejected(e):
                    temp = None
        return _sampled(_error_response(last_exc or RuntimeError("unknown")))

    system = LIVE_SYSTEM_PROMPT_BARE if layer == "layer1" else LEAKAGE_BOUND_PROMPT
    last_exc = None
    for _attempt in range(2):
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
                    **({} if temp is None else {"temperature": temp}),
                )
                return _sampled(_input_from_openai(response))
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=system,
                tools=[_open_submit_tool_anthropic()],
                tool_choice={"type": "tool", "name": SUBMIT_TOOL_NAME},
                messages=[{"role": "user", "content": _user_message(evidence)}],
                **({} if temp is None else {"temperature": temp}),
            )
            return _sampled(_input_from_anthropic(response))
        except Exception as e:  # noqa: BLE001 — SDK exception types vary
            last_exc = e
            if _param_rejected(e):
                temp = None
    return _sampled(_error_response(last_exc or RuntimeError("unknown")))


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


# Every record carries the provenance below so a published rate can be
# re-derived from the log alone, by someone who does not have the author, the
# shell history, or this session. §2.7 requires the ADAPTER commit pinned
# beside the harness commit; §5 requires the rest of the cell identity.
#
# Content hashes sit beside the commit hashes deliberately: a commit hash only
# pins a file that was committed, and measurement code is routinely run from a
# dirty tree. The sha256 of what was actually on disk at run time is the pin
# that cannot be wrong, and `dirty` says whether the two can disagree.
RECORD_SCHEMA_VERSION = 2
"""2 adds provenance/arm_id/strict/prompt_variant to every record.

Purely ADDITIVE: no channel definition, scoring rule, seed or filename
convention changed, so a v1 cell scores identically under the current scorer
and stays directly comparable. (No v1 cells survive on disk — see the run-log
note in the module docstring.)
"""


def _git_output(*args: str) -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _file_pin(path: str) -> dict[str, Any]:
    """Pin one source file: last commit that touched it, its content hash on
    disk, and whether the two can disagree."""
    commit = _git_output("log", "-1", "--format=%h", "--", path)
    status = _git_output("status", "--porcelain", "--", path)
    try:
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        digest = None
    return {"commit": commit, "sha256": digest, "dirty": bool(status)}


def run_provenance() -> dict[str, Any]:
    """What every record needs to be re-derivable without the author.

    Harness, adapter and the PRODUCT code under test are pinned separately:
    the contract arms route through ``investigate_leakage_bound``, so a rate
    that cannot name the leakage_investigator revision it measured cannot be
    reproduced.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    product = os.path.normpath(
        os.path.join(here, "..", "src", "mlcompass", "agents", "leakage_investigator.py")
    )
    try:
        import baseline_guardrails

        toolkit = baseline_guardrails.guardrails_version()
    except Exception:  # noqa: BLE001 — provenance must never abort a run
        toolkit = None
    import datetime

    return {
        "record_schema": RECORD_SCHEMA_VERSION,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": _git_output("rev-parse", "--short", "HEAD"),
        "harness": _file_pin(os.path.join(here, "reproduce_hallucination_ablation.py")),
        # §2.7: "the adapter commit hash is pinned in the run record beside the
        # harness commit". The Guardrails adapter is the adapter; the strict
        # -mode paths live in the harness and are pinned by the harness entry.
        "adapter": _file_pin(os.path.join(here, "baseline_guardrails.py")),
        "product": _file_pin(product),
        "guardrails_version": toolkit,
        "value_tolerance": VALUE_TOLERANCE,
    }


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
    meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run one cell (arm or sweep variant) with logging, resume, and latency.

    ``meta`` (provider/task/seed/evidence-hash) is stamped into every record
    so downstream table builders read cell identity from the DATA, not from a
    parse of the log filename (arm names contain the field separator).
    """
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
            log.append({**(meta or {}), "i": i, "arm": label, "model": model, **r, "scored": flags})
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
    strict: bool = False,
    temperature: float | None = 1.0,
    guardrails_reasks: int = 2,
    static_columns: tuple[str, ...] = STATIC_SCHEMA_COLUMNS,
) -> dict[str, list[dict[str, Any]]]:
    kind, client, default_model = _build_live_client(provider)
    use_model = model or default_model
    print(
        f"  provider={provider} kind={kind} model={use_model} strict={strict} "
        f"temperature={temperature}",
        file=sys.stderr,
    )
    if "static_schema_noenum" in arms:
        print(
            "  static_schema_noenum (A-STRICT-STATIC): strict mode, types and "
            "required fields, NO column enum — the column domain is "
            "unconstrained, so any invented name decodes successfully.",
            file=sys.stderr,
        )
    if "static_schema" in arms:
        coverage = _static_schema_coverage(
            list(static_columns), list(allowed), top_candidate(evidence)
        )
        print(
            f"  static_schema (A-STATIC-ENUM-STALE) domain: "
            f"{coverage['overlap']}/{coverage['n_evidence']} "
            f"of this task's evidence columns are in the frozen author-time list "
            f"(coverage {coverage['coverage'] * 100:.0f}%, anchor in list: "
            f"{coverage['anchor_in_static']})",
            file=sys.stderr,
        )
        if coverage["coverage"] >= 1.0:
            print(
                "  NOTE: the static domain covers this task exactly, so "
                "static_schema is IDENTICAL to tier_a+strict here by "
                "construction. It is a degenerate cell — the informative "
                "cells are the task instances the author-time list predates.",
                file=sys.stderr,
            )
        elif coverage["overlap"] == 0:
            print(
                "  WARNING: the static domain and this evidence are disjoint. "
                "Any fabrication rate this produces is a property of the "
                "mismatch, not of the mechanism — do not report it as a "
                "head-to-head result.",
                file=sys.stderr,
            )
    ctx = scorer_ctx or (set(allowed), evidence_correlation_map(evidence), top_candidate(evidence))
    ev_hash = _evidence_hash(evidence)
    provenance = run_provenance()
    print(
        f"  provenance: harness {provenance['harness']['commit']}"
        f"{'+dirty' if provenance['harness']['dirty'] else ''}, adapter "
        f"{provenance['adapter']['commit']}"
        f"{'+dirty' if provenance['adapter']['dirty'] else ''}, product "
        f"{provenance['product']['commit']}"
        f"{'+dirty' if provenance['product']['dirty'] else ''} "
        f"(pinned into every record)",
        file=sys.stderr,
    )
    out: dict[str, list[dict[str, Any]]] = {}
    for layer in arms:
        # Strict cells are DIFFERENT experimental cells (H4): distinct log name.
        arm_cell = f"{layer}+strict" if strict else layer
        log = (
            RunLog(
                log_dir,
                provider,
                use_model,
                task_label,
                arm_cell,
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
            arm_cell,
            lambda layer=layer: _live_one_response(  # type: ignore[misc]
                kind,
                client,
                use_model,
                layer,
                evidence,
                allowed,
                strict=strict,
                temperature=temperature,
                guardrails_reasks=guardrails_reasks,
                static_columns=static_columns,
            ),
            meta={
                "provider": provider,
                "task": task_label,
                "seed": seed,
                "evidence_hash": ev_hash,
                # Cell identity per §5: everything needed to re-derive this
                # cell's rate from the log alone.
                "arm_id": ARM_IDS.get(layer, "(unregistered arm)"),
                "strict": bool(strict),
                "prompt_variant": (
                    "bare"
                    if layer in ("layer1", "layer3_bare", "layer3_stress", "layer3_stress_generic")
                    or layer in STATIC_SCHEMA_ARMS
                    or layer in GUARDRAILS_ARMS
                    or layer == "tier_a"
                    else ("mechanical" if layer == "stress_mech" else "strict")
                ),
                "n_planned": n,
                "provenance": provenance,
            },
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
    temperature: float | None = 1.0,
) -> dict[str, list[dict[str, Any]]]:
    """Run the bare-prompt paraphrase sweep: same open tool, same evidence,
    same model — only the (rule-free) system prompt varies."""
    kind, client, default_model = _build_live_client(provider)
    use_model = model or default_model
    print(
        f"  sweep: provider={provider} model={use_model}, {len(SWEEP_BARE_VARIANTS)} variants, "
        f"temperature={temperature}",
        file=sys.stderr,
    )
    ctx = scorer_ctx or (set(allowed), evidence_correlation_map(evidence), top_candidate(evidence))

    def _one_sweep(prompt: str) -> dict[str, Any]:
        temp = temperature

        def _sampled(r: dict[str, Any]) -> dict[str, Any]:
            r["sampling"] = {"temperature": temp}
            return r

        last_exc: Exception | None = None
        for _attempt in range(2):
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
                        **({} if temp is None else {"temperature": temp}),
                    )
                    return _sampled(_input_from_openai(response))
                response = client.messages.create(
                    model=use_model,
                    max_tokens=1024,
                    system=prompt,
                    tools=[_open_submit_tool_anthropic()],
                    tool_choice={"type": "tool", "name": SUBMIT_TOOL_NAME},
                    messages=[{"role": "user", "content": _user_message(evidence)}],
                    **({} if temp is None else {"temperature": temp}),
                )
                return _sampled(_input_from_anthropic(response))
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if temp is not None and "temperature" in str(e).lower():
                    temp = None
        return _sampled(_error_response(last_exc or RuntimeError("unknown")))

    ev_hash = _evidence_hash(evidence)
    provenance = run_provenance()
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
            meta={
                "provider": provider,
                "task": task_label,
                "seed": seed,
                "evidence_hash": ev_hash,
                "arm_id": f"A-SWEEP-{name.upper()}",
                "strict": False,
                "prompt_variant": name,
                "n_planned": n,
                "provenance": provenance,
            },
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


def evidence_name_set(evidence: dict[str, Any]) -> set[str]:
    """Every string that appears anywhere in the evidence dict, key or value.

    Deliberately over-broad. Its only job is to answer one question -- could
    the narrator have read this name here? -- and for that a false positive
    (a name that happens to collide with an unrelated string in the evidence)
    is far cheaper than a false negative, which would report an invention
    that never happened.
    """
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                found.add(str(key))
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            found.add(node)

    walk(evidence)
    return found


def score_one(
    response: dict[str, Any],
    allowed_set: set[str],
    corr_map: dict[str, float],
    anchor: str | None,
    evidence_names: set[str] | None = None,
) -> dict[str, bool]:
    """Score a single response on the three contract channels.

    entity — references >= 1 column (in columns_referenced or a claim)
        outside the evidence set.

    When ``evidence_names`` is supplied, the entity channel is additionally
    split in two, because the 2026-09-15 baseline battery showed the blended
    rate does not mean what the paper calls it. Across 1,400 live responses
    every out-of-domain entity was ``r2`` -- the suspicious metric's own name,
    written into ``columns_referenced``. Nothing was invented.

    entity_invented  — a name that appears NOWHERE in the evidence dict. This
        is phantom-entity fabrication as the paper defines it: "a column the
        data does not contain".
    entity_misfiled  — a name that IS in the evidence but is not a column, put
        in a field that holds only columns. A real contract violation (a
        consumer that drops or correlates "r2" acts on something that is not
        a column) and NOT a hallucination.

    ``entity`` keeps its old meaning -- their disjunction -- so the contract's
    behaviour and every existing comparison are unchanged. What changes is
    that a reader can now see which of the two a rate is made of, instead of
    having to take a blended number on trust.
    value  — carries >= 1 claim whose column IS in the evidence but whose
        value differs from the measured one by > VALUE_TOLERANCE.
    omission — commits to a verdict other than cannot_determine yet never
        references the top-ranked candidate column. For contract arms the
        contract's own flag ("omitted") is authoritative.
    """
    cols = list(response.get("columns") or [])
    claims = [c for c in (response.get("claims") or []) if isinstance(c, dict)]
    claim_cols = [str(c.get("column", "")) for c in claims]
    outside = [c for c in cols + claim_cols if c not in allowed_set]
    entity = bool(outside)
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
    flags = {"entity": entity, "value": value, "omission": omission}
    if evidence_names is not None:
        flags["entity_invented"] = any(c not in evidence_names for c in outside)
        flags["entity_misfiled"] = any(c in evidence_names for c in outside)
    return flags


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
    "layer3_bare": "A-CONTRACT bare+TierA+TierB",
    "layer3_stress": "STRESS bare+TierB only",
    "layer3_stress_generic": "STRESS generic-retry",
    "tier_a": "TIER-A only (enum, no verify)",
    "stress_mech": "L3 + worst paraphrase",
    # Baseline comparison arms (analysis_plan §6, plan 2026-09 §2.1).
    "guardrails_stock": "BASE Guardrails (stock)",
    "guardrails_tierb": "BASE Guardrails (+TierB)",
    "static_schema_noenum": "BASE strict, no enum",
    "static_schema": "BASE strict, stale enum",
}

# The plan arm id for every harness arm (plan 2026-09 §2.1 + §9 A1).
#
# This mapping exists so a cell can never be run, logged or reported under a
# label naming a different hypothesis than the one it tests — the failure mode
# the pre-registration exists to prevent. `static_schema` (stale enum) and
# `static_schema_noenum` (no enum) are different arms with different rejecting
# outcomes, and only the latter is the registered A-STRICT-STATIC of §2.1. The
# id is stamped into every record and printed in the run plan.
ARM_IDS: dict[str, str] = {
    "layer1": "A-L1",
    "layer2": "A-L2",
    "layer3": "A-L3-SHIPPED",
    "layer3_bare": "A-CONTRACT",
    "layer3_stress": "A-STRESS",
    "layer3_stress_generic": "A-STRESS-GENERIC",
    "tier_a": "A-STRICT-ENUM (needs --strict)",
    "stress_mech": "A-L3-WORST-PARAPHRASE",
    "guardrails_stock": "A-GR-STOCK",
    "guardrails_tierb": "A-GR-OURS",
    "static_schema_noenum": "A-STRICT-STATIC",
    "static_schema": "A-STATIC-ENUM-STALE",
}

# The no-enforcement control for the baseline comparison. `layer1` already IS
# that control — bare prompt, open (enum-free) schema, no validation and no
# retry loop — and the baseline arms are built on exactly that call, so a
# separate control arm would duplicate it and pay twice for the same number.
NO_ENFORCEMENT_CONTROL_ARM = "layer1"

# The baseline battery: the registered §2.1 arm set runnable in ONE non-strict
# invocation — the no-enforcement floor, the contract comparator every contrast
# X1-X5 needs, and the mechanisms it is contrasted against. Same task, same
# evidence, same model, same 24-hour window (§2.7), same three channels; only
# the enforcement mechanism differs.
#
# Not in this tuple, and why:
#   A-STRICT-ENUM  — `tier_a` under the GLOBAL --strict flag, so it is a
#       separate invocation (--only-extra --strict) and a separate cell.
#   A-NEMO         — adapter not written; contrast X5 does not run this round.
# Both are reported as not run rather than quietly dropped.
BASELINE_ARMS: tuple[str, ...] = (
    NO_ENFORCEMENT_CONTROL_ARM,
    "layer3_bare",
    "layer3_stress",
    "guardrails_stock",
    "guardrails_tierb",
    "static_schema_noenum",
    "static_schema",
)

# The arms that carry a BASELINE mechanism (as opposed to the control and the
# contract arms that share the battery). Used to gate mock mode, which cannot
# run any of them.
MECHANISM_BASELINE_ARMS: frozenset[str] = frozenset(
    {"guardrails_stock", "guardrails_tierb", "static_schema", "static_schema_noenum"}
)

# Registered arms §2.1 lists that this harness cannot run today. Printed in the
# run plan, so an absent arm is a stated gap and never a silent one.
UNAVAILABLE_REGISTERED_ARMS: dict[str, str] = {
    "A-NEMO": (
        "NeMo Guardrails adapter not written; contrast X5 does not run this "
        "round and is reported as not run, not as a tie"
    ),
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
        "--strict",
        action="store_true",
        help=(
            "Emit tools with the provider's strict flag + strict-compatible "
            "schema (H4 enforcement dichotomy). Distinct experimental cells."
        ),
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "Sampling pin sent with every live call (default 1.0, the A3 "
            "cross-provider setting). If a provider rejects the parameter "
            "(reasoning endpoints), the pin is dropped for that cell and the "
            "record's sampling field says so. top_p is never sent."
        ),
    )
    ap.add_argument(
        "--pseudonymize",
        action="store_true",
        help=(
            "Hash every column name in the CSV task (A3.10 dataset-familiarity "
            "arm): the narrator sees content-hashed names, so memorized public "
            "schemas stop helping. Only valid with --task csv."
        ),
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
        choices=["synthetic", "synthetic-crowded", "csv", "case"],
        default="synthetic",
        help=(
            "Evidence source: the synthetic frame, the crowded value-channel "
            "stressor (frozen instance #14, analysis_plan A3.4), a real CSV "
            "with an injected leak (--csv-path/--target/--injector), or a "
            "real-world case study with a NATURAL documented leak (--case; "
            "analysis_plan A2)."
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
    ap.add_argument(
        "--only-baselines",
        action="store_true",
        help=(
            "Run ONLY the registered baseline battery (plan 2026-09 §2.1): "
            + ", ".join(BASELINE_ARMS)
            + ". A-STRICT-ENUM is a separate --only-extra --strict invocation; "
            "A-NEMO has no adapter. Live only."
        ),
    )
    ap.add_argument(
        "--guardrails-reasks",
        type=int,
        default=2,
        help=(
            "Reask budget for the Guardrails baseline arms (default 2, which "
            "is parity with the shipped contract's max_retries=2 — change it "
            "only for a deliberately registered loop-budget comparison)."
        ),
    )
    ap.add_argument(
        "--static-columns",
        default=None,
        help=(
            "Comma-separated override for the static_schema arm's author-time "
            "column domain. Default is the frozen STATIC_SCHEMA_COLUMNS (the "
            "reference task's evidence columns). Whatever is used is recorded "
            "in every response."
        ),
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

    if args.pseudonymize and args.task != "csv":
        raise SystemExit("--pseudonymize applies to --task csv only (A3.10 arm).")

    if args.task == "csv":
        if not args.csv_path or not args.target:
            raise SystemExit("--task csv requires --csv-path and --target.")
        evidence = build_csv_evidence(
            args.csv_path,
            args.target,
            seed=args.seed,
            injector=args.injector,
            pseudonymize=args.pseudonymize,
        )
        pseud = "/pseud" if args.pseudonymize else ""
        task_label = f"csv:{os.path.basename(args.csv_path)}/{args.target}/{args.injector}{pseud}"
    elif args.task == "case":
        if not args.case:
            raise SystemExit("--task case requires --case (bodyfat or sambanis).")
        if args.injector != "monotone_log":
            raise SystemExit("--injector does not apply to case studies (natural leaks).")
        from fetch_fabbench_datasets import build_case_evidence

        evidence = build_case_evidence(args.case)
        task_label = f"case:{args.case}"
    elif args.task == "synthetic-crowded":
        if args.injector != "monotone_log":
            raise SystemExit(
                "--injector does not apply to synthetic-crowded (frozen instance #14)."
            )
        evidence = build_synthetic_crowded_evidence(seed=args.seed)
        task_label = "synthetic_crowded"
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

    if log_dir:
        # The independent scorer (A3.9a) and the table builder (R8) re-derive
        # every number from raw logs + evidence; the evidence dict is dumped
        # next to the logs under its own content hash so cells and evidence
        # pair mechanically (log filenames embed the same hash).
        os.makedirs(log_dir, exist_ok=True)
        ev_path = os.path.join(log_dir, f"evidence_{_evidence_hash(evidence)}.json")
        with open(ev_path, "w", encoding="utf-8") as f:
            json.dump({"task_label": task_label, "evidence": evidence}, f, indent=2, default=str)
        print(f"  evidence dumped: {ev_path}", file=sys.stderr)

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
        if any(cell in ARM_IDS for cell in cells):
            # Every cell prints the registered arm id it will be logged under,
            # so a mislabelled cell is visible BEFORE it is paid for.
            for cell in cells:
                per_arm, arm_calls = estimate_cost_usd(use_model, (cell,), per_cell_n)
                cost = f"~${per_arm:.2f}" if per_arm is not None else "~$?"
                print(
                    f"    {cell:<22s} {ARM_IDS.get(cell, '(unregistered arm)'):<32s} "
                    f"{arm_calls:>4d} calls  {cost}",
                    file=sys.stderr,
                )
            if args.only_baselines:
                # Only the baseline battery is expected to cover the §2.1
                # arm set, so only there is a missing arm a gap worth naming.
                for arm_id, reason in UNAVAILABLE_REGISTERED_ARMS.items():
                    print(
                        f"    {'(not built)':<22s} {arm_id:<32s} NOT RUN: {reason}",
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
            temperature=args.temperature,
        )
        # A3.9b: transport errors are NOT data — exclude before scoring.
        rows = []
        for name, _ in SWEEP_BARE_VARIANTS:
            rs = sweep[name]
            valid = [r for r in rs if not r.get("error")]
            if len(valid) < len(rs):
                print(
                    f"  {name}: {len(rs) - len(valid)} transport error(s) excluded "
                    "per preregistration §7 (see run log for reasons)",
                    file=sys.stderr,
                )
            rows.append((name, score_responses(name, valid, allowed_set, corr_map, anchor)))
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
    # A3.2: STRESS runs BOTH retry variants (named + generic) by default.
    arms: tuple[str, ...] = (
        "layer1",
        "layer2",
        "layer3",
        "layer3_stress",
        "layer3_stress_generic",
    )
    if args.no_stress or args.mode == "mock":
        arms = ("layer1", "layer2", "layer3")
    if args.only_extra:
        if args.mode != "live":
            raise SystemExit("--only-extra is a live measurement; add --mode live.")
        arms = ("tier_a", "stress_mech")
    if args.only_baselines:
        if args.mode != "live":
            raise SystemExit("--only-baselines is a live measurement; add --mode live.")
        if args.only_extra:
            raise SystemExit("--only-baselines and --only-extra select different batteries.")
        arms = BASELINE_ARMS
    if args.smoke:
        if args.mode != "live":
            raise SystemExit("--smoke validates live wiring; add --mode live.")
        # A baseline smoke has to exercise the BASELINE code paths, otherwise
        # it proves the wiring of arms the run is not going to use.
        arms = BASELINE_ARMS if args.only_baselines else ("layer1", "layer3")

    # Baseline preflight — everything here is free and happens BEFORE the
    # cost banner, so a missing toolkit or a malformed domain list can never
    # abort a run that has already spent money.
    static_columns = STATIC_SCHEMA_COLUMNS
    if args.static_columns is not None:
        static_columns = tuple(c.strip() for c in args.static_columns.split(",") if c.strip())
        if not static_columns:
            raise SystemExit("--static-columns was given but parsed to an empty domain.")
    if "static_schema" in arms:
        # Whether A-STATIC-ENUM-STALE is informative on THIS task is decided by
        # the domain overlap, and a reader deciding whether to spend needs that
        # before the money leaves, not in the middle of the run. On the
        # reference task the frozen domain matches exactly, which makes the arm
        # a degenerate copy of tier_a+strict — worth knowing at plan time.
        pre = _static_schema_coverage(list(static_columns), list(allowed), anchor)
        print(
            f"  static_schema (A-STATIC-ENUM-STALE) domain vs this task: "
            f"{pre['overlap']}/{pre['n_evidence']} evidence columns covered "
            f"({pre['coverage'] * 100:.0f}%), anchor in domain: {pre['anchor_in_static']}",
            file=sys.stderr,
        )
        if pre["coverage"] >= 1.0:
            print(
                "  NOTE: the frozen domain covers this task exactly, so "
                "A-STATIC-ENUM-STALE is IDENTICAL to A-STRICT-ENUM here by "
                "construction — a DEGENERATE cell. The stale-enum arm is only "
                "informative on a task the author-time list predates. Contrast "
                "X3b from this cell would compare an arm against itself.",
                file=sys.stderr,
            )
        elif pre["overlap"] == 0:
            print(
                "  WARNING: the frozen domain and this evidence are disjoint. "
                "Any rate that produces is a property of the mismatch, not of "
                "the mechanism — do not report it as a head-to-head result.",
                file=sys.stderr,
            )
    if any(arm in GUARDRAILS_ARMS for arm in arms):
        import baseline_guardrails

        if not baseline_guardrails.guardrails_available():
            raise SystemExit(
                "The selected arms include a Guardrails baseline but the "
                "toolkit is not installed:\n"
                "    pip install 'guardrails-ai==0.11.0'\n"
                "(the 'baselines' extra in pyproject.toml). No API calls made."
            )
        print(
            f"  baseline toolkit: guardrails-ai "
            f"{baseline_guardrails.guardrails_version()}, reask budget "
            f"{args.guardrails_reasks} (contract parity: max_retries=2)",
            file=sys.stderr,
        )
    if args.mode == "mock":
        if any(arm in MECHANISM_BASELINE_ARMS for arm in arms):
            raise SystemExit("The baseline arms are live measurements; add --mode live.")
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
            strict=args.strict,
            temperature=args.temperature,
            guardrails_reasks=args.guardrails_reasks,
            static_columns=static_columns,
        )

    # A3.9b: transport errors are NOT data — exclude from N, report per cell.
    valid_by_layer: dict[str, list[dict[str, Any]]] = {}
    error_counts: dict[str, int] = {}
    for layer in arms:
        rs = responses_by_layer[layer]
        valid_by_layer[layer] = [r for r in rs if not r.get("error")]
        error_counts[layer] = len(rs) - len(valid_by_layer[layer])
        if error_counts[layer]:
            print(
                f"  {layer}: {error_counts[layer]} transport error(s) excluded "
                "per preregistration §7 (see run log for reasons)",
                file=sys.stderr,
            )

    results = [
        score_responses(layer, valid_by_layer[layer], allowed_set, corr_map, anchor)
        for layer in arms
    ]

    # Tier B catch telemetry — how often the contract demonstrably fired.
    rejection_summary: dict[str, dict[str, int]] = {}
    for layer in arms:
        rej = [int(r.get("rejections") or 0) for r in valid_by_layer[layer]]
        rejection_summary[layer] = {
            "responses_with_catches": sum(1 for x in rej if x > 0),
            "total_catches": sum(rej),
            "errors_excluded": error_counts[layer],
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
