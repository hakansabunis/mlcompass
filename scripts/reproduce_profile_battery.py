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
    # Review 2026-09-30: the verifier against a well-specified configuration.
    # Both arms send the shipped rule-bearing prompt, the call-time enums and
    # field descriptions on every reference-bearing field (STRONG_DESCRIPTIONS);
    # they differ only in whether the payload is verified.
    "strong_noverify": (True, False, PROFILE_BOUND_PROMPT, "P-STRONG-NOVERIFY"),
    "strong_verify": (True, True, PROFILE_BOUND_PROMPT, "P-STRONG-CONTRACT"),
}

STRONG_ARMS = ("strong_noverify", "strong_verify")
STRONG_DESCRIPTIONS = {
    "columns_referenced": (
        "Names of dataset columns from the profile that you cite. "
        "Column names only, not the names of statistics."
    ),
    "column": "A dataset column name from the profile.",
    "statistic": "A statistic the profile reports for that column.",
    "value": "The number exactly as the profile reports it for this column and statistic.",
}


def _described_schema(original):
    """Wrap build_payload_schema so every reference-bearing field is described."""

    def wrapped(*a, **k):
        schema = original(*a, **k)
        props = schema["properties"]
        props["columns_referenced"]["description"] = STRONG_DESCRIPTIONS["columns_referenced"]
        claim = props["claims"]["items"]["properties"]
        for field in ("column", "statistic", "value"):
            if field in claim:
                claim[field]["description"] = STRONG_DESCRIPTIONS[field]
        return schema

    return wrapped

PROVIDERS = {
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "deepseek-chat"),
    "openai": ("OPENAI_API_KEY", None, "gpt-5.4-mini"),
    "mistral": ("MISTRAL_API_KEY", "https://api.mistral.ai/v1", "ministral-8b-latest"),
    "ollama": ("OLLAMA_NO_KEY", "http://localhost:11434/v1", "qwen2.5:7b"),
    # The product default of the profile narrator (PROFILE_MODEL_DEFAULT).
    "anthropic": ("ANTHROPIC_API_KEY", None, "claude-opus-4-7"),
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


# Review item P0-4 (DA C2, EIC W3): 10 of the 11 misfiled values on the
# 24-column frame put cat_feature_07's number on num_feature_07, a pair our own
# generator named with a shared suffix. With --natural-names the SAME data (same
# RNG draws, same order) is written under distinct names that share no suffix or
# stem across the numeric and categorical blocks, so a misfiling that survives
# is not an artefact of our naming.
_NATURAL_NUMERIC = (
    "age", "income", "tenure_months", "balance", "credit_score", "monthly_spend",
    "login_count", "session_minutes", "support_tickets", "page_views",
    "days_since_purchase", "discount_rate", "basket_size", "referral_count",
)
_NATURAL_CATEGORICAL = (
    "region", "plan_tier", "device_type", "acquisition_channel",
    "payment_method", "language", "industry",
)


def build_profile_evidence(
    seed: int = 0,
    n_rows: int = 1200,
    frame_columns: int | None = None,
    natural_names: bool = False,
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
        if natural_names:
            name = _NATURAL_NUMERIC[i - 1] if i <= len(_NATURAL_NUMERIC) else f"metric_{i:03d}"
        else:
            name = f"num_feature_{i:02d}"
        cols[name] = v
    for i in range(1, 8):
        k = 3 + i * 4
        c = rng.choice([f"cat{j}" for j in range(k)], n_rows).astype(object)
        c[rng.random(n_rows) < 0.03 * i] = None
        cols[_NATURAL_CATEGORICAL[i - 1] if natural_names else f"cat_feature_{i:02d}"] = c
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


def interleave_orders(seed: int, n: int) -> list[tuple[str, str]]:
    """For each index, the order in which the two naming schemes are called.

    Analysis plan A9: the naming comparison runs both schemes in one run, one
    call of each per index, in an order drawn at random per index, so that
    drift within the run and the harness commit are shared by both schemes.
    """
    import random  # noqa: PLC0415

    rng = random.Random(f"interleave-{seed}")
    orders = []
    for _ in range(n):
        pair = ["suffixed", "natural"]
        rng.shuffle(pair)
        orders.append((pair[0], pair[1]))
    return orders


def evidence_hash(evidence: dict[str, Any]) -> str:
    import hashlib

    blob = json.dumps(evidence, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def make_client(provider: str) -> tuple[Any, str]:
    env, base, model = PROVIDERS[provider]
    key = os.environ.get(env) or ("ollama" if provider == "ollama" else None)
    if not key:
        raise SystemExit(
            f"{env} is not set. Run `source scripts/load_keys.sh` first."
        )
    if provider == "anthropic":
        import anthropic

        return anthropic.Anthropic(api_key=key), model
    from openai import OpenAI

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
    ap.add_argument("--resume", action="store_true",
                    help="Continue an interrupted arm, skipping indices already logged.")
    ap.add_argument("--natural-names", action="store_true",
                    help="Same frame, distinct column names with no shared suffix "
                         "(review item P0-4). Use a separate --log-dir.")
    ap.add_argument("--interleave-arms", action="store_true",
                    help="Analysis plan A11: run exactly two --arm values in one run, one "
                         "call of each per index, in a random order per index; one log file "
                         "per arm. Use a fresh --log-dir.")
    ap.add_argument("--interleave-names", action="store_true",
                    help="Analysis plan A9: run the suffixed and the natural-name "
                         "frame in one run, one call of each per index, in a random "
                         "order per index; one log file per scheme. Use a fresh --log-dir.")
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

    if args.interleave_names:
        return run_interleaved(args)
    if args.interleave_arms:
        return run_interleaved_arms(args)

    arms = args.arm or ["bare"]
    evidence = build_profile_evidence(
        seed=args.seed, frame_columns=args.frame_columns, natural_names=args.natural_names
    )
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

    import mlcompass.agents.profile_narrator as _pn  # noqa: PLC0415

    plain_schema = _pn.build_payload_schema
    for arm in arms:
        enforce, do_verify, prompt, arm_id = ARMS[arm]
        _pn.build_payload_schema = (
            _described_schema(plain_schema) if arm in STRONG_ARMS else plain_schema
        )
        out = args.log_dir / (
            f"{args.provider}_{model.replace(':', '-')}_profile_{arm}"
            f"_n{args.n}_seed{args.seed}_e{ehash}.jsonl"
        )
        done: set[int] = set()
        if out.exists():
            if not args.resume:
                print(f"{arm}: {out.name} exists — refusing to overwrite paid responses "
                      "(pass --resume to continue it)")
                continue
            done = {
                int(json.loads(line)["i"])
                for line in out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            print(f"{arm}: resuming, {len(done)} responses already logged")

        flagged = {"entity": 0, "value": 0, "omission": 0}
        rows = []
        t0 = time.time()
        for i in range(args.n):
            if i in done:
                continue
            started = time.time()
            try:
                res = narrate_profile_bound(
                    evidence,
                    client=client,
                    model=model,
                    provider="anthropic" if args.provider == "anthropic" else "openai",
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
                    "frame_names": "natural" if args.natural_names else "suffixed",
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


def _record(args, arm, arm_id, model, ehash, i, scheme, res, scored, schema_bytes, started,
            prov, extra) -> dict[str, Any]:
    return {
        "provider": args.provider, "task": "profile", "arm": arm, "arm_id": arm_id,
        "model": model, "evidence_hash": ehash, "i": i, "seed": args.seed,
        "n_planned": args.n, "max_columns": args.max_columns,
        "frame_columns": args.frame_columns, "frame_names": scheme,
        "columns": res["columns_referenced"], "claims": res["claims"],
        "raw_columns": res["raw_columns_referenced"], "raw_claims": res["raw_claims"],
        "attempts": res["attempts"], "narration": res["narration"],
        "verdict": res["verdict"], "confidence": res["confidence"],
        "omitted": res["omitted_critical_evidence"], "rejections": res["schema_rejections"],
        "rejection_kinds": res["rejection_kinds"], "provider_calls": res["attempts_made"],
        "latency_ms": round((time.time() - started) * 1000, 1), "scored": scored,
        "admissible_columns": res["admissible_columns"],
        "admissible_statistics": res["admissible_statistics"],
        "schema_bytes": schema_bytes, "sampling": {"temperature": args.temperature},
        "provenance": prov, **extra,
    }


def interleave_arm_orders(seed: int, n: int, arms: list[str]) -> list[tuple[str, str]]:
    """For each index, the order in which the two arms are called (A11)."""
    import random  # noqa: PLC0415

    rng = random.Random(f"interleave-arms-{seed}")
    orders = []
    for _ in range(n):
        pair = list(arms)
        rng.shuffle(pair)
        orders.append((pair[0], pair[1]))
    return orders


def run_interleaved_arms(args) -> int:
    """Analysis plan A11: two arms on one frame, interleaved, one harness commit."""
    import mlcompass.agents.profile_narrator as _pn  # noqa: PLC0415

    arms = args.arm or []
    if len(arms) != 2:
        raise SystemExit("--interleave-arms needs exactly two --arm values.")
    ev = build_profile_evidence(seed=args.seed, frame_columns=args.frame_columns,
                                natural_names=args.natural_names)
    trimmed = compact(ev, max_columns=args.max_columns)
    bound = PROFILE.bind(trimmed)
    ehash = evidence_hash(trimmed)
    scheme = "natural" if args.natural_names else "suffixed"
    orders = interleave_arm_orders(args.seed, args.n, arms)
    print(f"{scheme}: evidence {ehash}; first orders {orders[:5]}")
    if args.dry_run:
        return 0
    client, model = make_client(args.provider)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / f"evidence_{ehash}.json").write_text(
        json.dumps({"task_label": "profile", "evidence": trimmed}, indent=1, default=str),
        encoding="utf-8")
    prov = {**provenance(), "interleaved_arms": list(arms)}
    plain_schema = _pn.build_payload_schema
    outs = {a: args.log_dir / (f"{args.provider}_{model.replace(':', '-')}_profile_{a}_interleaved"
                               f"_n{args.n}_seed{args.seed}_e{ehash}.jsonl") for a in arms}
    done: dict[str, set[int]] = {a: set() for a in arms}
    if any(o.exists() for o in outs.values()):
        if not args.resume:
            print("interleaved logs exist, refusing to overwrite paid responses (pass --resume)")
            return 0
        for a, o in outs.items():
            if o.exists():
                done[a] = {int(json.loads(line)["i"]) for line in
                           o.read_text(encoding="utf-8").splitlines() if line.strip()}
        print(f"resuming, {min(len(d) for d in done.values())} complete pairs logged")
    for i, order in enumerate(orders):
        for pos, arm in enumerate(order):
            if i in done[arm]:
                continue
            enforce, do_verify, prompt, arm_id = ARMS[arm]
            _pn.build_payload_schema = (
                _described_schema(plain_schema) if arm in STRONG_ARMS else plain_schema)
            schema_bytes = len(json.dumps(_pn.build_submit_tool_openai(bound)))
            started = time.time()
            try:
                res = narrate_profile_bound(
                    ev, client=client, model=model,
                    provider="anthropic" if args.provider == "anthropic" else "openai",
                    max_retries=2 if do_verify else 0, system_prompt=prompt,
                    enforce_schema_enum=enforce, verify_response=do_verify,
                    temperature=args.temperature, max_columns=args.max_columns)
            except Exception as e:  # noqa: BLE001
                print(f"  {arm} {i}: transport error {type(e).__name__}: {str(e)[:90]}")
                continue
            v = verify({"verdict": res["verdict"], "columns_referenced": res["columns_referenced"],
                        "claims": res["claims"]}, bound, PROFILE)
            scored = {"entity": bool(v.entity), "value": bool(v.value),
                      "omission": bool(res["omitted_critical_evidence"])}
            rec = _record(args, arm, arm_id, model, ehash, i, scheme, res, scored, schema_bytes,
                          started, prov, {"interleave": {"pair": i, "position": pos}})
            with outs[arm].open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{args.n} pairs", flush=True)
    return 0


def run_interleaved(args) -> int:
    """Analysis plan A9: both naming schemes, interleaved, one harness commit."""
    import mlcompass.agents.profile_narrator as _pn  # noqa: PLC0415

    schemes = {}
    for scheme in ("suffixed", "natural"):
        ev = build_profile_evidence(seed=args.seed, frame_columns=args.frame_columns,
                                    natural_names=scheme == "natural")
        trimmed = compact(ev, max_columns=args.max_columns)
        bound = PROFILE.bind(trimmed)
        schemes[scheme] = (ev, trimmed, bound, evidence_hash(trimmed))
        print(f"{scheme}: evidence {schemes[scheme][3]}, {len(bound.values)} measured quantities, "
              f"anchor={bound.anchor!r}")
    orders = interleave_orders(args.seed, args.n)
    print(f"first orders: {orders[:5]}")
    if args.dry_run:
        return 0
    client, model = make_client(args.provider)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    for _, trimmed, _, ehash in schemes.values():
        (args.log_dir / f"evidence_{ehash}.json").write_text(
            json.dumps({"task_label": "profile", "evidence": trimmed}, indent=1, default=str),
            encoding="utf-8")
    prov = {**provenance(), "interleaved_names": True}
    plain_schema = _pn.build_payload_schema
    for arm in args.arm or ["tier_a"]:
        enforce, do_verify, prompt, arm_id = ARMS[arm]
        _pn.build_payload_schema = (
            _described_schema(plain_schema) if arm in STRONG_ARMS else plain_schema)
        outs = {s: args.log_dir / (f"{args.provider}_{model.replace(':', '-')}_profile_{arm}_{s}"
                                   f"_interleaved_n{args.n}_seed{args.seed}_e{schemes[s][3]}.jsonl")
                for s in schemes}
        done: dict[str, set[int]] = {s: set() for s in schemes}
        if any(o.exists() for o in outs.values()):
            if not args.resume:
                print(f"{arm}: interleaved logs exist, refusing to overwrite paid responses "
                      "(pass --resume to continue them)")
                continue
            for s_, o in outs.items():
                if o.exists():
                    done[s_] = {int(json.loads(line)["i"]) for line in
                                o.read_text(encoding="utf-8").splitlines() if line.strip()}
            print(f"{arm}: resuming, {min(len(d) for d in done.values())} complete pairs logged")
        for i, order in enumerate(orders):
            for pos, scheme in enumerate(order):
                if i in done[scheme]:
                    continue
                ev, _, bound, ehash = schemes[scheme]
                schema_bytes = len(json.dumps(_pn.build_submit_tool_openai(bound)))
                started = time.time()
                try:
                    res = narrate_profile_bound(
                        ev, client=client, model=model,
                        provider="anthropic" if args.provider == "anthropic" else "openai",
                        max_retries=2 if do_verify else 0, system_prompt=prompt,
                        enforce_schema_enum=enforce, verify_response=do_verify,
                        temperature=args.temperature, max_columns=args.max_columns)
                except Exception as e:  # noqa: BLE001
                    print(f"  {arm} {i} {scheme}: transport error {type(e).__name__}: {str(e)[:90]}")
                    continue
                v = verify({"verdict": res["verdict"], "columns_referenced": res["columns_referenced"],
                            "claims": res["claims"]}, bound, PROFILE)
                scored = {"entity": bool(v.entity), "value": bool(v.value),
                          "omission": bool(res["omitted_critical_evidence"])}
                rec = _record(args, arm, arm_id, model, ehash, i, scheme, res, scored, schema_bytes,
                              started, prov, {"interleave": {"pair": i, "position": pos}})
                with outs[scheme].open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, default=str) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  {arm}: {i + 1}/{args.n} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
