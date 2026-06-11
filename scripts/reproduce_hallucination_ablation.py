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
                --provider (anthropic / deepseek / openai) and set its API key
                env var (ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY).
                **This is the mode that produces the paper's Table I** — the
                certified 2026-06-12 battery (paper Tables I-II) used --provider deepseek.

  --mode mock   — ILLUSTRATIVE ONLY, no API calls. A deterministic simulator for
                grading and for users without a key. The per-layer rates are
                placeholders, and Layer 3's 0% here is true *by construction of
                the simulator* — it is NOT a measurement. Do not cite mock-mode
                output as a result; use --mode live for that.

Usage::

    # real measurement, N=200 per layer (paper Table I)
    python scripts/reproduce_hallucination_ablation.py --mode live --n 200

    # tighter CIs at N=1000
    python scripts/reproduce_hallucination_ablation.py --mode live --n 1000

    # no-API illustrative demo (clearly labelled as such)
    python scripts/reproduce_hallucination_ablation.py --mode mock

The ``--json`` flag emits a machine-readable summary for a regression harness.
See ``docs/THREAT_MODEL.md`` and ``docs/KNOWN_LIMITATIONS.md`` for what the
contract does and does not promise.
"""

from __future__ import annotations

import argparse
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

from mlcompass.agents.leakage_investigator import (  # noqa: E402
    LEAKAGE_BOUND_PROMPT,
    LEAKAGE_MODEL_DEFAULT,
    SUBMIT_TOOL_NAME,
    VALUE_TOLERANCE,
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


def build_csv_evidence(csv_path: str, target: str, seed: int = 0) -> dict[str, Any]:
    """Build evidence from a REAL third-party dataset with an injected leak.

    Loads the CSV, injects a monotone log-of-target leak plus near-perfect
    predictions (the same controlled failure as the synthetic frame), and runs
    the shipped ``detect_leakage``. The feature distributions, column names,
    and dataset shape are all external — answering the "everything is
    self-designed" critique with a real-data replication task.
    """
    import numpy as np
    import pandas as pd

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
    df[f"log_{target}_leak"] = np.log(y_arr - y_arr.min() + 1.0) + rng.normal(0, 0.01, n)
    df["y_pred"] = y_arr + rng.normal(0, max(1e-9, 0.001 * y_arr.std()), n)

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


# Provider table: which API key, base URL, and default model each uses. The
# DeepSeek endpoint is OpenAI-compatible; its weaker schema enforcement makes
# Tier B (deterministic validation) the load-bearing guarantee — a useful
# cross-provider robustness signal for the contract.
PROVIDERS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "kind": "anthropic",
        "key_env": "ANTHROPIC_API_KEY",
        "model": LEAKAGE_MODEL_DEFAULT,
        "base_url": None,
    },
    "deepseek": {
        "kind": "openai",
        "key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
    },
    "openai": {
        "kind": "openai",
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
        "base_url": None,
    },
}


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


def _normalize(tool_input: dict[str, Any]) -> dict[str, Any]:
    cols = tool_input.get("columns_referenced") or []
    claims = [c for c in (tool_input.get("claims") or []) if isinstance(c, dict)]
    return {
        "columns": [str(c) for c in cols] if isinstance(cols, list) else [],
        "claims": claims,
        "verdict": str(tool_input.get("verdict", "")),
        "omitted": None,  # computed by the scorer for layers 1-2
        "rejections": 0,  # Tier B catches; nonzero only on contract arms
    }


def _input_from_anthropic(response: Any) -> dict[str, Any]:
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            raw = getattr(block, "input", {})
            return _normalize(dict(raw) if isinstance(raw, dict) else {})
    return _normalize({})


def _input_from_openai(response: Any) -> dict[str, Any]:
    message = response.choices[0].message
    calls = getattr(message, "tool_calls", None) or []
    if not calls:
        return _normalize({})
    args = calls[0].function.arguments
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
    except (json.JSONDecodeError, TypeError):
        return _normalize({})
    return _normalize(parsed if isinstance(parsed, dict) else {})


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
    if layer in ("layer3", "layer3_stress"):
        stress = layer == "layer3_stress"
        result = investigate_leakage_bound(
            evidence,
            client=client,
            model=model,
            provider=("openai" if kind == "openai" else "anthropic"),
            system_prompt=(LIVE_SYSTEM_PROMPT_BARE if stress else None),
            enforce_schema_enum=not stress,
        )
        return {
            "columns": list(result["columns_referenced"]),
            "claims": list(result["claims"]),
            "verdict": result["verdict"],
            "omitted": bool(result["omitted_critical_evidence"]),
            "rejections": int(result["schema_rejections"]),
        }

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


def live_run(
    provider: str,
    n: int,
    evidence: dict[str, Any],
    allowed: list[str],
    model: str | None = None,
    arms: tuple[str, ...] = ("layer1", "layer2", "layer3", "layer3_stress"),
) -> dict[str, list[dict[str, Any]]]:
    kind, client, default_model = _build_live_client(provider)
    use_model = model or default_model
    print(f"  provider={provider} kind={kind} model={use_model}", file=sys.stderr)
    out: dict[str, list[dict[str, Any]]] = {}
    for layer in arms:
        responses: list[dict[str, Any]] = []
        for i in range(n):
            responses.append(_live_one_response(kind, client, use_model, layer, evidence, allowed))
            if (i + 1) % 25 == 0:
                print(f"  {layer}: {i + 1}/{n}", file=sys.stderr)
        out[layer] = responses
    return out


def sweep_run(
    provider: str, n: int, evidence: dict[str, Any], allowed: list[str], model: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Run the bare-prompt paraphrase sweep: same open tool, same evidence,
    same model — only the (rule-free) system prompt varies."""
    kind, client, default_model = _build_live_client(provider)
    use_model = model or default_model
    print(
        f"  sweep: provider={provider} model={use_model}, {len(SWEEP_BARE_VARIANTS)} variants",
        file=sys.stderr,
    )
    out: dict[str, list[dict[str, Any]]] = {}
    for name, prompt in SWEEP_BARE_VARIANTS:
        responses: list[dict[str, Any]] = []
        for i in range(n):
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
                    responses.append(_input_from_openai(response))
                else:
                    response = client.messages.create(
                        model=use_model,
                        max_tokens=1024,
                        system=prompt,
                        tools=[_open_submit_tool_anthropic()],
                        tool_choice={"type": "tool", "name": SUBMIT_TOOL_NAME},
                        messages=[{"role": "user", "content": _user_message(evidence)}],
                    )
                    responses.append(_input_from_anthropic(response))
            except Exception:  # noqa: BLE001
                responses.append(_normalize({}))
            if (i + 1) % 25 == 0:
                print(f"  {name}: {i + 1}/{n}", file=sys.stderr)
        out[name] = responses
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


def score_responses(
    layer: str,
    responses: list[dict[str, Any]],
    allowed_set: set[str],
    corr_map: dict[str, float],
    anchor: str | None,
) -> LayerResult:
    """Score one layer's responses on the three contract channels.

    entity-fab — response references >= 1 column (in columns_referenced or a
        claim) outside the evidence set.
    value-fab  — response carries >= 1 claim whose column IS in the evidence
        but whose value differs from the measured one by > VALUE_TOLERANCE.
    omission   — response commits to a verdict other than cannot_determine yet
        never references the top-ranked candidate column. For layer 3 the
        contract's own flag is used (the contract retries on omission; the
        flag marks what survived the budget).
    """
    n = len(responses)
    k_entity = k_value = k_omit = 0
    for r in responses:
        cols = list(r.get("columns") or [])
        claims = [c for c in (r.get("claims") or []) if isinstance(c, dict)]
        claim_cols = [str(c.get("column", "")) for c in claims]
        if any(c not in allowed_set for c in cols + claim_cols):
            k_entity += 1
        for c in claims:
            col = str(c.get("column", ""))
            val = c.get("value")
            if col in corr_map and (
                not isinstance(val, (int, float))
                or abs(float(val) - corr_map[col]) > VALUE_TOLERANCE
            ):
                k_value += 1
                break
        if r.get("omitted") is not None:
            k_omit += 1 if r["omitted"] else 0
        else:
            committed = bool(cols or claims) and r.get("verdict") not in ("", "cannot_determine")
            if committed and anchor is not None and anchor not in set(cols) | set(claim_cols):
                k_omit += 1
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
        help="Live provider: 'anthropic', 'deepseek' (OpenAI-compatible), or 'openai'.",
    )
    ap.add_argument("--model", default=None, help="Override the provider's default model.")
    ap.add_argument(
        "--task",
        choices=["synthetic", "csv"],
        default="synthetic",
        help="Evidence source: the synthetic frame, or a real CSV (--csv-path/--target).",
    )
    ap.add_argument("--csv-path", default=None, help="Real dataset CSV for --task csv.")
    ap.add_argument("--target", default=None, help="Numeric target column for --task csv.")
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
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = ap.parse_args()

    if args.task == "csv":
        if not args.csv_path or not args.target:
            raise SystemExit("--task csv requires --csv-path and --target.")
        evidence = build_csv_evidence(args.csv_path, args.target, seed=args.seed)
        task_label = f"csv:{os.path.basename(args.csv_path)}/{args.target}"
    else:
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

    # ---------------- Sweep mode: bare-prompt paraphrase distribution -------- #
    if args.sweep:
        if args.mode != "live":
            raise SystemExit("--sweep is a live measurement; add --mode live.")
        sweep = sweep_run(args.provider, args.n, evidence, allowed, model=args.model)
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

    if args.mode == "mock":
        responses_by_layer = mock_run(args.n, allowed, corr_map, anchor, seed=args.seed)
    else:
        responses_by_layer = live_run(
            args.provider, args.n, evidence, allowed, model=args.model, arms=arms
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

    if args.json:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "provider": args.provider if args.mode == "live" else None,
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
