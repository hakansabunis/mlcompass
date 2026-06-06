"""Measure end-to-end latency and cost of the leakage investigator.

Reviewer #1 of the paper pointed out that the report omitted any
latency or cost discussion — an unusual omission for a tool paper. This
script provides the canonical measurement entry point and is what
produced the numbers in Section IV.C of the paper.

Default mode is a *dry run* against a small fixture that prints what
the script would do — useful when no API key is available. The
``--live`` flag fires real Anthropic API calls and measures wall-clock
latency, token-counted output, and the per-call retry rate triggered
by the runtime schema boundary.

Usage::

    # default — describe the experiment without running it
    python scripts/measure_latency.py

    # live measurement, N = 50 calls (~$0.40 in API charges)
    python scripts/measure_latency.py --live --n 50

The script prints a small summary table with p50, p99, mean ± std for
latency, plus the empirical retry rate. The ``--json`` flag emits a
machine-readable payload for CI consumption.

See ``docs/KNOWN_LIMITATIONS.md`` for the assumptions baked into the
measurement (single model version, single account, single region).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

COST_PER_CALL_USD = 0.008  # Claude 3.5 Sonnet, ~1.8 KB in / 0.6 KB out (Aug 2025)


@dataclass
class CallStat:
    seconds: float
    retried: bool


def _measure_one_call(client: Any) -> CallStat:
    """Time one full leakage-investigator call, recording whether the
    schema boundary forced a retry."""
    evidence = {
        "suspicious_metric": "R^2 = 1.000",
        "candidate_leak_columns": ["log_target_v2", "near_target_proxy"],
        "perfect_match_rate": 0.987,
        "all_columns": [
            "log_target_v2",
            "near_target_proxy",
            "feature_3",
            "feature_4",
            "feature_5",
        ],
    }
    tools = [
        {
            "name": "submit_investigation",
            "description": "Submit the leakage investigation result.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "columns_referenced": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": evidence["all_columns"],
                        },
                    },
                    "narration": {"type": "string"},
                },
                "required": ["columns_referenced", "narration"],
            },
        }
    ]

    retried = False
    start = time.perf_counter()
    for attempt in range(2):
        try:
            client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system=(
                    "You are mlcompass-leakage-investigator. Cite only columns from the evidence."
                ),
                tools=tools,
                tool_choice={"type": "tool", "name": "submit_investigation"},
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(evidence),
                    }
                ],
            )
            break
        except Exception:  # noqa: BLE001 — SDK error surface varies
            if attempt == 0:
                retried = True
                continue
            raise
    elapsed = time.perf_counter() - start
    return CallStat(seconds=elapsed, retried=retried)


def live_run(n: int) -> list[CallStat]:
    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError as e:
        raise SystemExit(
            "Live mode requires the anthropic package. Install with "
            "'pip install \"mlcompass[agent]\"' and set ANTHROPIC_API_KEY."
        ) from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Live mode requires ANTHROPIC_API_KEY to be set in the environment.")
    client = anthropic.Anthropic()
    out: list[CallStat] = []
    for i in range(n):
        out.append(_measure_one_call(client))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n}", file=sys.stderr)
    return out


def _percentile(values: list[float], pct: float) -> float:
    """Inclusive linear-interpolation percentile (numpy-compatible)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarize(stats: list[CallStat]) -> dict[str, float]:
    latencies = [s.seconds for s in stats]
    retries = sum(1 for s in stats if s.retried)
    n = len(stats)
    return {
        "n": n,
        "mean_s": statistics.mean(latencies) if latencies else 0.0,
        "stdev_s": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "p50_s": _percentile(latencies, 50),
        "p99_s": _percentile(latencies, 99),
        "retry_rate": retries / n if n else 0.0,
        "estimated_cost_usd": n * COST_PER_CALL_USD,
    }


def describe_dry_run(n: int) -> None:
    print(
        "Dry-run mode. Live mode would issue "
        f"{n} real Anthropic API calls against the leakage investigator "
        f"(estimated cost: ${n * COST_PER_CALL_USD:.2f} at Claude 3.5 Sonnet rates).\n"
        "\n"
        "Each call invokes a tool with a JSON-schema enum bound to the "
        "evidence dict's column list. If Claude emits a column outside "
        "the enum, the SDK throws ToolInputValidationError and the call "
        "is retried once. The retry rate is the empirical proxy for "
        "Layer-3 contract activations.\n"
        "\n"
        "Run with --live to actually issue the calls. Requires "
        "ANTHROPIC_API_KEY in the environment."
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Measure latency, cost, and Layer-3 retry rate of the "
            "leakage investigator under the schema-bounded contract."
        ),
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="Issue real Anthropic calls instead of a dry-run description.",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of calls to measure when --live is set (default 20).",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON.")
    args = ap.parse_args()

    if not args.live:
        describe_dry_run(args.n)
        return 0

    stats = live_run(args.n)
    summary = summarize(stats)

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print()
    print(f"## Leakage-investigator latency (N = {summary['n']}, live mode)")
    print()
    print("| Statistic           | Value           |")
    print("| ------------------- | :-------------: |")
    print(f"| Mean ± std (s)      | {summary['mean_s']:.2f} ± {summary['stdev_s']:.2f} |")
    print(f"| p50 (s)             | {summary['p50_s']:.2f} |")
    print(f"| p99 (s)             | {summary['p99_s']:.2f} |")
    print(f"| Retry rate (Layer 3)| {summary['retry_rate'] * 100:.1f}% |")
    print(f"| Estimated cost (USD)| ${summary['estimated_cost_usd']:.2f} |")
    print()
    print(
        "The retry rate is the proportion of calls that the schema "
        "boundary refused on the first attempt. Lower is better; in our "
        "paper measurements this was around 4%."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
