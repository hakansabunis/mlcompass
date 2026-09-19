"""The second evidence-closed task, measured under the same contract.

Two independent reviewers made the same objection: the paper defines a class of
tasks and evaluates one member, so the class-level claim rests on an argument
rather than on evidence. This is the second member.

It narrates `mlcompass.tools.dataset.analyze_dataset` --- a deterministic
profiler shipping in the same tool --- through
`mlcompass.agents.evidence_contract`, the same verifier the leakage contract
uses, with no branch anywhere naming either task.

Why this shape and not another. A profile differs from leakage in the two ways
that matter to the paper's claims:

  * the entity domain is every column in the frame, not a threshold-filtered
    candidate list, so it is as wide as the data --- which makes this also the
    natural place to measure what Tier A costs as the enum grows;
  * thirteen statistics per column against leakage's one, and which exist
    depends on the column, so the value table must be keyed on the pair and the
    statistic domain must be computed per call.

Arms, named to match the leakage battery exactly so a difference between the
two tasks is a difference in the tasks:

  bare        no faithfulness rule, open schema, no verification    (A-L1)
  tier_a      evidence-bound enums, NO verification                 (A-TIER-A-ONLY)
  contract    Tier A + Tier B, bare prompt                          (A-CONTRACT)
  stress      Tier B only, enums removed                            (A-STRESS)

Scoring is the same three channels, applied to what reaches the user, by the
same `verify()` the contract uses. For the enforced arms that is the repaired
response, which is the point: the measure is what the engineer sees.

    source scripts/load_keys.sh
    python scripts/reproduce_profile_battery.py --arm bare --n 20 --provider deepseek
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlcompass.agents.evidence_contract import PROFILE, verify  # noqa: E402
from mlcompass.agents.profile_narrator import (  # noqa: E402
    PROFILE_BARE_PROMPT,
    PROFILE_BOUND_PROMPT,
    compact,
    narrate_profile_bound,
)

RUNS = ROOT / "scripts" / "runs" / "profile"

ARMS = {
    # arm: (enforce_enum, verify, prompt, plan arm id)
    "bare": (False, False, PROFILE_BARE_PROMPT, "P-L1"),
    "tier_a": (True, False, PROFILE_BARE_PROMPT, "P-TIER-A-ONLY"),
    "contract": (True, True, PROFILE_BARE_PROMPT, "P-CONTRACT"),
    "stress": (False, True, PROFILE_BARE_PROMPT, "P-STRESS"),
}

PROVIDERS = {
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "deepseek-chat"),
    "openai": ("OPENAI_API_KEY", None, "gpt-5.4-mini"),
    "mistral": ("MISTRAL_API_KEY", "https://api.mistral.ai/v1", "ministral-8b-latest"),
    "ollama": ("OLLAMA_NO_KEY", "http://localhost:11434/v1", "qwen2.5:7b"),
}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def build_profile_evidence(
    seed: int = 0, n_rows: int = 1200, frame_columns: int | None = None
) -> dict[str, Any]:
    """A wide frame with real data-quality problems, profiled by the shipped tool.

    The evidence the experiment narrates is produced by `analyze_dataset`, not
    written by hand, for the same reason the leakage battery calls
    `detect_leakage`: an experiment that narrates a hand-built dictionary
    measures a shape the product never emits.

    The frame is deliberately ordinary. Nothing here is constructed to trip a
    narrator: mixed types, missingness that rises across the categorical block,
    a few injected outliers, a datetime and a binary target. The trap that
    exists --- thirteen statistic names sitting beside two dozen column names
    in one dictionary --- is a property of what a profiler emits, not something
    we planted.
    """
    import numpy as np
    import pandas as pd

    from mlcompass.tools.dataset import analyze_dataset

    rng = np.random.default_rng(seed)
    cols: dict[str, Any] = {}

    # The scalability knob. The 24-column frame is the task; a wider one is the
    # same task with a wider admissible set, which is the only variable Section
    # 12 concedes we never moved. Extra columns are numeric and structurally
    # identical to the first fourteen, so what changes between widths is the
    # size of the enum and nothing else about the narration problem.
    extra = 0 if frame_columns is None else max(0, frame_columns - 24)
    for i in range(1, 15 + extra):
        v = rng.normal(50 + i, 12, n_rows)
        v[rng.random(n_rows) < (0.02 * (i % 5))] = np.nan
        if i % 4 == 0:
            v[rng.integers(0, n_rows, 8)] = 9999.0
        cols[f"num_feature_{i:02d}"] = v
    for i in range(1, 8):
        k = 3 + i * 4
        c = rng.choice([f"cat{j}" for j in range(k)], n_rows).astype(object)
        c[rng.random(n_rows) < 0.03 * i] = None
        cols[f"cat_feature_{i:02d}"] = c
    cols["is_active"] = rng.random(n_rows) > 0.4
    cols["signup_date"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 700, n_rows), unit="D"
    )
    cols["target"] = (rng.random(n_rows) > 0.7).astype(int)

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d, "frame.csv")
        pd.DataFrame(cols).to_csv(path, index=False)
        return analyze_dataset(path, target_column="target")


def evidence_hash(evidence: dict[str, Any]) -> str:
    import hashlib

    blob = json.dumps(evidence, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def make_client(provider: str) -> tuple[Any, str]:
    env, base, model = PROVIDERS[provider]
    from openai import OpenAI

    key = os.environ.get(env) or ("ollama" if provider == "ollama" else None)
    if not key:
        raise SystemExit(
            f"{env} is not set. Run `source scripts/load_keys.sh` first."
        )
    return OpenAI(api_key=key, base_url=base) if base else OpenAI(api_key=key), model


def provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return "unknown"

    return {
        "record_schema": 3,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "repo_commit": git("rev-parse", "--short", "HEAD"),
        "tree_dirty": bool(git("status", "--porcelain")),
        "contract": "evidence_contract.PROFILE",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", choices=sorted(ARMS), default=None)
    ap.add_argument("--provider", default="deepseek", choices=sorted(PROVIDERS))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-columns", type=int, default=None,
                    help="Truncate the enum after profiling. Cannot exceed the frame.")
    ap.add_argument("--frame-columns", type=int, default=None,
                    help="Build a frame this wide. The Tier A scalability knob: "
                         "10/50/100/500/1000 gives the enum cost curve.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--log-dir", type=pathlib.Path, default=RUNS)
    ap.add_argument("--dry-run", action="store_true",
                    help="Bind and print the domains, spend nothing.")
    ap.add_argument("--scale-table", action="store_true",
                    help="Emit the LaTeX rows of the Tier A cost table and exit. "
                         "No API call: a serialised schema is a property of the "
                         "contract, not of a provider on a date.")
    args = ap.parse_args()

    if args.scale_table:
        from mlcompass.agents.profile_narrator import build_submit_tool_openai
        print("% Generated: python scripts/reproduce_profile_battery.py --scale-table")
        for width in (10, 24, 50, 100, 250, 500, 1000):
            ev = compact(
                build_profile_evidence(seed=args.seed, frame_columns=width),
                max_columns=width,
            )
            bd = PROFILE.bind(ev)
            sch = len(json.dumps(build_submit_tool_openai(bd)))
            evb = len(json.dumps(ev, default=str))
            print(
                rf"{width:<5} & {sch / 1000:5.1f}\,kB & {evb / 1000:6.1f}\,kB "
                rf"& {100 * sch / (sch + evb):.0f}\,\% \\"
            )
        return 0

    arms = args.arm or ["bare"]
    evidence = build_profile_evidence(seed=args.seed, frame_columns=args.frame_columns)
    trimmed = compact(evidence, max_columns=args.max_columns)
    bound = PROFILE.bind(trimmed)
    ehash = evidence_hash(trimmed)

    prompt_bytes = len(json.dumps(trimmed, default=str))
    schema_bytes = len(
        json.dumps(
            __import__("mlcompass.agents.profile_narrator", fromlist=["x"])
            .build_submit_tool_openai(bound)
        )
    )
    print(
        f"evidence {ehash}: {len(bound.domains['columns_referenced'])} columns, "
        f"{len(bound.domains['claims[].statistic'])} statistics, "
        f"{len(bound.values)} measured quantities, anchor={bound.anchor!r}"
    )
    print(f"prompt {prompt_bytes} bytes, tool schema {schema_bytes} bytes")
    if args.dry_run:
        return 0

    client, model = make_client(args.provider)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / f"evidence_{ehash}.json").write_text(
        json.dumps({"task_label": "profile", "evidence": trimmed}, indent=1, default=str),
        encoding="utf-8",
    )
    prov = provenance()

    for arm in arms:
        enforce, do_verify, prompt, arm_id = ARMS[arm]
        out = args.log_dir / (
            f"{args.provider}_{model.replace(':', '-')}_profile_{arm}"
            f"_n{args.n}_seed{args.seed}_e{ehash}.jsonl"
        )
        if out.exists():
            print(f"{arm}: {out.name} exists — refusing to overwrite paid responses")
            continue

        flagged = {"entity": 0, "value": 0, "omission": 0}
        rows = []
        t0 = time.time()
        for i in range(args.n):
            started = time.time()
            try:
                res = narrate_profile_bound(
                    evidence,
                    client=client,
                    model=model,
                    provider="openai",
                    max_retries=2 if do_verify else 0,
                    # Every arm gets the bare prompt. Giving the contract arm
                    # the strict one would confound the mechanism with the
                    # instruction, which is the confound the leakage battery's
                    # A-CONTRACT was defined to remove: the shipped enforcement
                    # stack measured with the faithfulness prompt taken out, so
                    # it is comparable to bare-prompt baselines. An earlier
                    # version of this line handed `contract` PROFILE_BOUND_PROMPT
                    # and reintroduced it.
                    system_prompt=prompt,
                    enforce_schema_enum=enforce,
                    verify_response=do_verify,
                    temperature=args.temperature,
                    max_columns=args.max_columns,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  {arm} {i}: transport error {type(e).__name__}: {str(e)[:90]}")
                continue

            # Score what reaches the user, on all three channels, with the same
            # verifier. For the enforced arms this is the repaired response.
            seen = {
                "verdict": res["verdict"],
                "columns_referenced": res["columns_referenced"],
                "claims": res["claims"],
            }
            v = verify(seen, bound, PROFILE)
            scored = {
                "entity": bool(v.entity),
                "value": bool(v.value),
                "omission": bool(res["omitted_critical_evidence"]),
            }
            for k in flagged:
                flagged[k] += int(scored[k])

            rows.append(
                {
                    "provider": args.provider,
                    "task": "profile",
                    "arm": arm,
                    "arm_id": arm_id,
                    "model": model,
                    "evidence_hash": ehash,
                    "i": i,
                    "seed": args.seed,
                    "n_planned": args.n,
                    "max_columns": args.max_columns,
                    "frame_columns": args.frame_columns,
                    "columns": res["columns_referenced"],
                    "claims": res["claims"],
                    "raw_columns": res["raw_columns_referenced"],
                    "raw_claims": res["raw_claims"],
                    "attempts": res["attempts"],
                    "narration": res["narration"],
                    "verdict": res["verdict"],
                    "confidence": res["confidence"],
                    "omitted": res["omitted_critical_evidence"],
                    "rejections": res["schema_rejections"],
                    "rejection_kinds": res["rejection_kinds"],
                    "provider_calls": res["attempts_made"],
                    "latency_ms": round((time.time() - started) * 1000, 1),
                    "scored": scored,
                    "admissible_columns": res["admissible_columns"],
                    "admissible_statistics": res["admissible_statistics"],
                    "schema_bytes": schema_bytes,
                    "sampling": {"temperature": args.temperature},
                    "provenance": prov,
                }
            )
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rows[-1], default=str) + "\n")

            if (i + 1) % 10 == 0:
                print(f"  {arm}: {i + 1}/{args.n}")

        n = len(rows)
        print(f"\n{arm_id} ({arm}), N={n}, {time.time() - t0:.0f}s -> {out.name}")
        for channel in ("entity", "value", "omission"):
            k = flagged[channel]
            lo, hi = wilson(k, n)
            print(f"  {channel:9s} {k:3d}/{n:<4d} {100 * k / n if n else 0:5.1f}% "
                  f"[{lo:.1f}, {hi:.1f}]")
        catches = sum(r["rejections"] for r in rows)
        print(f"  Tier B catches: {catches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
