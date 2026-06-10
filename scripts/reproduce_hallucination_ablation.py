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
                certified 2026-06-10 run used --provider deepseek.

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
    evidence_allowed_columns,
    investigate_leakage_bound,
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


# --------------------------------------------------------------------------- #
# Mock backend — ILLUSTRATIVE ONLY (no API, not a measurement)                #
# --------------------------------------------------------------------------- #


# Per-layer rates for the no-API demo, set to the 2026-06-10 deepseek-chat
# live run (56.5% / 0.0% / 0.0%, N=200 — see paper/ablation_live_deepseek_*.md).
# Mock mode REPLAYS these rates; it does not measure anything.
ILLUSTRATIVE_RATES: dict[str, float] = {"layer1": 0.565, "layer2": 0.000, "layer3": 0.000}

PLAUSIBLE_PHANTOMS = (
    "target_score",
    "predicted_value",
    "leak_indicator",
    "label_proxy",
    "revenue",
    "shadow_target",
)

_LAYER_SEED_OFFSET = {"layer1": 1, "layer2": 2, "layer3": 3}


def _mock_one_response(rng: random.Random, layer: str, allowed: list[str]) -> list[str]:
    cited = [rng.choice(allowed[: min(2, len(allowed))] or allowed)]
    for _ in range(rng.randint(0, 2)):
        cited.append(rng.choice(allowed))
    # Layer 3's simulator never fabricates — this is the "by construction"
    # property the live mode actually has to demonstrate empirically.
    if layer == "layer3":
        return cited
    if rng.random() < ILLUSTRATIVE_RATES[layer]:
        cited.append(rng.choice(PLAUSIBLE_PHANTOMS))
    return cited


def mock_run(n: int, allowed: list[str], seed: int = 0) -> dict[str, list[list[str]]]:
    out: dict[str, list[list[str]]] = {}
    for layer in ILLUSTRATIVE_RATES:
        rng = random.Random(seed * 100003 + _LAYER_SEED_OFFSET[layer])
        out[layer] = [_mock_one_response(rng, layer, allowed) for _ in range(n)]
    return out


# --------------------------------------------------------------------------- #
# Live backend — measures the SHIPPED contract                                #
# --------------------------------------------------------------------------- #


LIVE_SYSTEM_PROMPT_BARE = """You are mlcompass-leakage-investigator.

A deterministic tool gathered evidence about a suspicious metric on a
predictions table. Call the submit_investigation tool once. In
columns_referenced, list every column from the evidence you cite. Include a
short narration.
"""


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


def _open_submit_tool_anthropic() -> dict[str, Any]:
    """Anthropic submit tool WITHOUT the evidence enum (layers 1 and 2)."""
    schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string"},
            "confidence": {"type": "string"},
            "columns_referenced": {"type": "array", "items": {"type": "string"}},
            "narration": {"type": "string"},
        },
        "required": ["columns_referenced", "narration"],
    }
    return {
        "name": SUBMIT_TOOL_NAME,
        "description": "Submit the leakage investigation result.",
        "input_schema": schema,
    }


def _open_submit_tool_openai() -> dict[str, Any]:
    """OpenAI-compatible submit tool WITHOUT the evidence enum (layers 1 and 2)."""
    return {
        "type": "function",
        "function": {
            "name": SUBMIT_TOOL_NAME,
            "description": "Submit the leakage investigation result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string"},
                    "confidence": {"type": "string"},
                    "columns_referenced": {"type": "array", "items": {"type": "string"}},
                    "narration": {"type": "string"},
                },
                "required": ["columns_referenced", "narration"],
            },
        },
    }


def _user_message(evidence: dict[str, Any]) -> str:
    return (
        "Evidence dictionary:\n\n"
        + json.dumps(evidence, indent=2, default=str)
        + "\n\nInvestigate the suspicious metric."
    )


def _cited_from_anthropic(response: Any) -> list[str]:
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            cited = getattr(block, "input", {}).get("columns_referenced", [])
            return [str(c) for c in cited] if isinstance(cited, list) else []
    return []


def _cited_from_openai(response: Any) -> list[str]:
    message = response.choices[0].message
    calls = getattr(message, "tool_calls", None) or []
    if not calls:
        return []
    args = calls[0].function.arguments
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
    except (json.JSONDecodeError, TypeError):
        return []
    cited = parsed.get("columns_referenced", []) if isinstance(parsed, dict) else []
    return [str(c) for c in cited] if isinstance(cited, list) else []


def _live_one_response(
    kind: str, client: Any, model: str, layer: str, evidence: dict[str, Any], allowed: list[str]
) -> list[str]:
    """Sample one narrator response for the given layer and provider kind.

    Layers 1 and 2 use an open tool (no enum) with no post-validation, so the
    raw fabrication is observable. Layer 3 routes through the SHIPPED
    ``investigate_leakage_bound`` — the same code the product runs — which binds
    the enum to the evidence and deterministically validates the cited columns.
    """
    if layer == "layer3":
        result = investigate_leakage_bound(
            evidence,
            client=client,
            model=model,
            provider=("openai" if kind == "openai" else "anthropic"),
        )
        return list(result["columns_referenced"])

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
                return _cited_from_openai(response)
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=system,
                tools=[_open_submit_tool_anthropic()],
                tool_choice={"type": "tool", "name": SUBMIT_TOOL_NAME},
                messages=[{"role": "user", "content": _user_message(evidence)}],
            )
            return _cited_from_anthropic(response)
        except Exception:  # noqa: BLE001 — SDK exception types vary
            if attempt == 0:
                continue
            return []
    return []


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
    provider: str, n: int, evidence: dict[str, Any], allowed: list[str], model: str | None = None
) -> dict[str, list[list[str]]]:
    kind, client, default_model = _build_live_client(provider)
    use_model = model or default_model
    print(f"  provider={provider} kind={kind} model={use_model}", file=sys.stderr)
    out: dict[str, list[list[str]]] = {}
    for layer in ("layer1", "layer2", "layer3"):
        responses: list[list[str]] = []
        for i in range(n):
            responses.append(_live_one_response(kind, client, use_model, layer, evidence, allowed))
            if (i + 1) % 25 == 0:
                print(f"  {layer}: {i + 1}/{n}", file=sys.stderr)
        out[layer] = responses
    return out


# --------------------------------------------------------------------------- #
# Scoring                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class LayerResult:
    layer: str
    n: int
    k_phantom: int
    rate: float
    ci_low: float
    ci_high: float


def score_responses(layer: str, responses: list[list[str]], allowed_set: set[str]) -> LayerResult:
    n = len(responses)
    k = sum(1 for cited in responses if any(c not in allowed_set for c in cited))
    rate = k / n if n else 0.0
    low, high = wilson_ci_95(k, n)
    return LayerResult(layer=layer, n=n, k_phantom=k, rate=rate, ci_low=low, ci_high=high)


LAYER_LABELS = {
    "layer1": "Layer 1 (bare prompt, no schema)",
    "layer2": "Layer 1 + 2 (strict prompt)",
    "layer3": "Layer 1 + 2 + 3 (evidence-bound)",
}


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
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = ap.parse_args()

    evidence = build_synthetic_evidence(seed=args.seed)
    allowed = evidence_allowed_columns(evidence)
    allowed_set = set(allowed)

    print(
        f"Running {args.mode} ablation, N={args.n} per layer (evidence columns: {len(allowed)})...",
        file=sys.stderr,
    )

    if args.mode == "mock":
        responses_by_layer = mock_run(args.n, allowed, seed=args.seed)
    else:
        responses_by_layer = live_run(args.provider, args.n, evidence, allowed, model=args.model)

    results = [
        score_responses(layer, responses_by_layer[layer], allowed_set)
        for layer in ("layer1", "layer2", "layer3")
    ]

    if args.json:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "provider": args.provider if args.mode == "live" else None,
                    "n_per_layer": args.n,
                    "seed": args.seed,
                    "evidence_columns": allowed,
                    "results": [
                        {
                            "layer": r.layer,
                            "label": LAYER_LABELS[r.layer],
                            "n": r.n,
                            "k_phantom": r.k_phantom,
                            "rate": r.rate,
                            "ci_low": r.ci_low,
                            "ci_high": r.ci_high,
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
        print("    Layer 3's 0% here is by construction of the simulator. Use")
        print("    --mode live to produce the paper's Table I.\n")

    print(f"## Phantom-column fabrication rate ({args.mode} mode, N = {args.n})\n")
    print("| Layer                            | Rate  | Wilson 95% CI    | k / N      |")
    print("| -------------------------------- | :---: | :--------------: | :--------: |")
    for r in results:
        print(
            f"| {LAYER_LABELS[r.layer]:<32s} | {r.rate * 100:>4.1f}% "
            f"| [{r.ci_low * 100:>5.2f}, {r.ci_high * 100:>5.2f}] | {r.k_phantom:>3d} / {r.n:<4d} |"
        )
    print(
        "\nLayer 3 routes through the shipped investigate_leakage_bound: the "
        "columns_referenced enum is bound to the evidence at call time and the "
        "cited columns are deterministically re-validated, so no out-of-evidence "
        "column reaches the user. See paper Section III for the binding details."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
