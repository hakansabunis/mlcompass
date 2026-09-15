"""Execute and score benchmark runs per `protocol.md`.

One invocation runs every (dataset x configuration x repetition) cell,
preserves the evidence each run produced, scores it against the frozen
registry in `ground_truth.json`, and appends one row per run to
`results.csv`.

    python benchmark/run_benchmark.py --plan                 # freeze a plan, run nothing
    python benchmark/run_benchmark.py --config deterministic # run and score
    python benchmark/run_benchmark.py --dry-run              # print the cells

Protocol points this implements, and where they bite:

- **Plan before execution** (section 1). `--plan` writes `runs/<experiment_id>/plan.md`
  and stops. Execution refuses to start unless a plan for that experiment id
  already exists, so a run cannot be scored under settings invented after
  seeing it.
- **Identical inputs** (section 2). Every dataset is re-hashed against
  `ground_truth.json` before use; a mismatch aborts rather than silently
  measuring a different file.
- **Fresh workspace per repetition** (section 3.2). Each run gets its own
  temp directory with no `.mlcompass/`, so nothing carries over.
- **Monotonic timing** (section 3.4) from command start to completion,
  excluding preparation.
- **Failures are kept** (section 3.6). A failed or timed-out run is recorded
  with blank detection counts, never discarded or silently retried.

Scoring is deliberately partial. `known_issues`, `correct_detections`,
`missed_issues` and `unverified_findings` are computed from frozen match
rules. `false_positives` and `hallucinations` are left blank: refuting a
reported issue takes judgement this script does not have, and section 4
says an unrefutable claim belongs in `unverified_findings` rather than
being counted as a false positive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCH = Path(__file__).parent
GROUND_TRUTH = BENCH / "ground_truth.json"
RESULTS = BENCH / "results.csv"
RUNS = BENCH / "runs"

PROTOCOL_VERSION = "1.0"
DEFAULT_TIMEOUT = 600

# Configurations under comparison. Each names the mlcompass command and the
# arguments that make it that configuration; `{csv}` and `{target}` are
# substituted per dataset. Keep these frozen: changing a configuration after
# a scored run has used it makes the rows incomparable.
CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "deterministic": {
        "description": "advise with no LLM layer — the rule-based detectors alone.",
        "argv": ["advise", "{csv}", "--target", "{target}"],
        "uses_llm": False,
    },
    "llm": {
        "description": "advise with the LLM advisor on top of the same deterministic analysis.",
        "argv": ["advise", "{csv}", "--target", "{target}", "--llm"],
        "uses_llm": True,
    },
}

RESULT_COLUMNS = [
    "run_id",
    "experiment_id",
    "protocol_version",
    "dataset_id",
    "case_id",
    "configuration_id",
    "repeat_index",
    "seed",
    "started_at_utc",
    "mlcompass_commit",
    "provider",
    "model",
    "status",
    "known_issues",
    "correct_detections",
    "missed_issues",
    "false_positives",
    "unverified_findings",
    "hallucinations",
    "runtime_seconds",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "estimated_llm_cost_usd",
    "artifacts_path",
    "notes",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=BENCH.parent, timeout=30
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _load_ground_truth() -> dict[str, Any]:
    return json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))


def _verify_inputs(gt: dict[str, Any]) -> list[str]:
    """Re-hash every pinned dataset. Returns a list of problems."""
    problems = []
    for d in gt["datasets"]:
        path = BENCH / d["csv_path"]
        if not path.exists():
            problems.append(f"{d['dataset_id']}: missing {d['csv_path']}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != d["sha256"]:
            problems.append(
                f"{d['dataset_id']}: sha256 mismatch\n"
                f"    registry {d['sha256']}\n    on disk  {actual}"
            )
    return problems


# --------------------------------------------------------------------------- #
# Scoring                                                                     #
# --------------------------------------------------------------------------- #


def _split_findings(output: str) -> list[str]:
    """Pull the bulleted findings out of a rendered report.

    `protocol.md` section 4 asks for compound reports to be split into
    distinct claims and deduplicated. mlcompass renders one warning per
    bullet, so the bullet is the claim boundary; rich wraps long lines, so
    a continuation line (no bullet marker, indented) folds into the claim
    above it rather than becoming a claim of its own.
    """
    findings: list[str] = []
    for raw in output.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith(("•", "-", "*")) and len(stripped) > 2:
            findings.append(stripped.lstrip("•-* ").strip())
        elif findings and line.startswith(" ") and stripped:
            findings[-1] += " " + stripped
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for f in findings:
        key = re.sub(r"\s+", " ", f.lower())
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def score(output: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Match a run's output against the frozen cases for its dataset."""
    findings = _split_findings(output)
    haystack = re.sub(r"\s+", " ", output.lower())

    detected: list[str] = []
    matched_findings: set[int] = set()
    for case in cases:
        pattern = case["match_pattern"].lower()
        if not re.search(pattern, haystack):
            continue
        if not all(str(v).lower() in haystack for v in case.get("required_values", [])):
            continue
        detected.append(case["issue_id"])
        for i, f in enumerate(findings):
            flat = re.sub(r"\s+", " ", f.lower())
            if re.search(pattern, flat):
                matched_findings.add(i)

    unverified = [f for i, f in enumerate(findings) if i not in matched_findings]
    return {
        "known_issues": len(cases),
        "correct_detections": len(detected),
        "missed_issues": len(cases) - len(detected),
        "detected_ids": detected,
        "missed_ids": [c["issue_id"] for c in cases if c["issue_id"] not in detected],
        "unverified_findings": len(unverified),
        "unverified_texts": unverified,
        "all_findings": findings,
    }


# --------------------------------------------------------------------------- #
# Execution                                                                   #
# --------------------------------------------------------------------------- #


def execute(
    dataset: dict[str, Any],
    config_id: str,
    run_dir: Path,
    *,
    timeout: int,
    seed: int,
) -> dict[str, Any]:
    """Run one cell in a fresh workspace and preserve its evidence."""
    config = CONFIGURATIONS[config_id]
    csv_path = (BENCH / dataset["csv_path"]).resolve()
    argv = [a.format(csv=str(csv_path), target=dataset["target"]) for a in config["argv"]]
    command = [sys.executable, "-X", "utf8", "-m", "mlcompass.cli", *argv]

    workspace = Path(tempfile.mkdtemp(prefix="mlcbench_"))
    started = _utc_now()
    clock = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=workspace,
            timeout=timeout,
        )
        runtime = time.monotonic() - clock
        status = "completed" if proc.returncode == 0 else "failed"
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        runtime = time.monotonic() - clock
        status, code = "timeout", None
        stdout = exc.stdout.decode("utf-8", "replace") if exc.stdout else ""
        stderr = exc.stderr.decode("utf-8", "replace") if exc.stderr else ""
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    (run_dir / "command.json").write_text(
        json.dumps(
            {
                "command": command,
                "cwd": "fresh temp workspace, no .mlcompass present",
                "started_at_utc": started,
                "exit_code": code,
                "status": status,
                "runtime_seconds": round(runtime, 3),
                "seed": seed,
                "input_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "mlcompass_commit": _git("rev-parse", "HEAD"),
                "tree_dirty": bool(_git("status", "--porcelain")),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": status,
        "stdout": stdout,
        "stderr": stderr,
        "runtime": runtime,
        "started": started,
    }


def write_scoring(
    run_dir: Path, dataset: dict[str, Any], result: dict[str, Any], scored: dict[str, Any] | None
) -> None:
    lines = [
        f"# Scoring — {run_dir.name}",
        "",
        f"Dataset: {dataset['dataset_id']} ({dataset['name']})",
        f"Status: {result['status']}",
        "",
    ]
    if scored is None:
        lines += [
            "Run did not complete; detection counts are left blank per protocol.md section 4.",
            "",
        ]
    else:
        lines += [
            f"known_issues: {scored['known_issues']}",
            f"correct_detections: {scored['correct_detections']}  {scored['detected_ids']}",
            f"missed_issues: {scored['missed_issues']}  {scored['missed_ids']}",
            f"unverified_findings: {scored['unverified_findings']}",
            "",
            "## Findings reported by the tool",
            "",
        ]
        lines += [f"- {f}" for f in scored["all_findings"]] or ["- (none)"]
        lines += ["", "## Unmatched findings (unverified, not refuted)", ""]
        lines += [f"- {f}" for f in scored["unverified_texts"]] or ["- (none)"]
        lines += [
            "",
            "false_positives and hallucinations are blank: refuting a reported",
            "issue needs a reviewer, and protocol.md section 4 puts an unrefutable",
            "claim in unverified_findings rather than in false_positives.",
        ]
    (run_dir / "scoring.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_result(row: dict[str, Any]) -> None:
    exists = RESULTS.exists() and RESULTS.read_text(encoding="utf-8").strip()
    with RESULTS.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULT_COLUMNS})


def write_plan(
    experiment_id: str,
    gt: dict[str, Any],
    config_ids: list[str],
    repeats: int,
    timeout: int,
    seeds: list[int],
) -> Path:
    plan_dir = RUNS / experiment_id
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_dir / "plan.md"
    cells = [
        f"| {d['dataset_id']} | {c} | {repeats} | {', '.join(map(str, seeds))} |"
        for d in gt["datasets"]
        for c in config_ids
    ]
    plan.write_text(
        "\n".join(
            [
                f"# Experiment plan — {experiment_id}",
                "",
                f"Frozen at {_utc_now()} (UTC), before execution, per protocol.md section 1.",
                "",
                f"- Protocol version: {PROTOCOL_VERSION}",
                f"- Ground-truth registry: ground_truth.json"
                f" v{gt['registry_version']}, frozen {gt['frozen_at_utc']}",
                f"- mlcompass commit: {_git('rev-parse', 'HEAD')}",
                f"- Tree dirty at planning time: {bool(_git('status', '--porcelain'))}",
                f"- Python {platform.python_version()} on {platform.platform()}",
                f"- Timeout per run: {timeout}s",
                f"- Repetitions per cell: {repeats}; seeds {seeds}",
                f"- Reviewers: {gt['reviewers']}",
                "",
                "## Configurations",
                "",
                *[f"- `{c}`: {CONFIGURATIONS[c]['description']}" for c in config_ids],
                "",
                "## Cells",
                "",
                "| dataset | configuration | repeats | seeds |",
                "| --- | --- | --- | --- |",
                *cells,
                "",
                "## Scoring",
                "",
                gt["matching"]["rule"],
                "",
                f"Scope: {gt['matching']['scope']}",
                "",
                "false_positives and hallucinations are not scored automatically; see",
                "ground_truth.json `out_of_scope` for why.",
                "",
                "## Execution order",
                "",
                "Sequential, dataset-major then configuration, on the recorded environment.",
                "Every repetition starts in a fresh temp workspace with no `.mlcompass/`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-id", default=None, help="reuse an existing frozen plan")
    ap.add_argument(
        "--config",
        action="append",
        choices=sorted(CONFIGURATIONS),
        help="configuration to run (repeatable; default: deterministic)",
    )
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--seed", type=int, action="append", help="seed per repetition (repeatable)")
    ap.add_argument("--plan", action="store_true", help="freeze a plan and exit without running")
    ap.add_argument("--dry-run", action="store_true", help="print the cells and exit")
    ap.add_argument("--provider", default="none")
    ap.add_argument("--model", default="none")
    args = ap.parse_args()

    gt = _load_ground_truth()
    config_ids = args.config or ["deterministic"]
    seeds = args.seed or list(range(args.repeats))
    if len(seeds) < args.repeats:
        seeds = (seeds * args.repeats)[: args.repeats]

    problems = _verify_inputs(gt)
    if problems:
        print("Input verification failed (protocol.md section 2):")
        for p in problems:
            print(f"  {p}")
        print("\nRun: python benchmark/fetch_datasets.py --verify")
        return 1
    print(f"Inputs verified: {len(gt['datasets'])} dataset(s) match the registry.\n")

    if args.dry_run:
        for d in gt["datasets"]:
            for c in config_ids:
                print(
                    f"  {d['dataset_id']:<14} {c:<14}"
                    f" x{args.repeats}  known_issues={len(d['cases'])}"
                )
        return 0

    experiment_id = (
        args.experiment_id or f"exp-{datetime.now(timezone.utc):%Y%m%d}-{uuid.uuid4().hex[:6]}"
    )
    plan_path = RUNS / experiment_id / "plan.md"

    if args.plan:
        path = write_plan(experiment_id, gt, config_ids, args.repeats, args.timeout, seeds)
        print(f"Plan frozen: {path.relative_to(BENCH)}")
        print(
            f"\nRun it with:\n  python benchmark/run_benchmark.py --experiment-id {experiment_id} "
            + " ".join(f"--config {c}" for c in config_ids)
        )
        return 0

    if not plan_path.exists():
        print(f"No frozen plan at {plan_path.relative_to(BENCH)}.")
        print("protocol.md section 1 requires the plan before execution. Freeze one with --plan.")
        return 1
    print(f"Using frozen plan: {plan_path.relative_to(BENCH)}\n")

    uses_llm = any(CONFIGURATIONS[c]["uses_llm"] for c in config_ids)
    if uses_llm and args.provider == "none":
        print(
            "An --llm configuration needs --provider and --model recorded (protocol.md section 1)."
        )
        return 1

    total = 0
    for dataset in gt["datasets"]:
        for config_id in config_ids:
            for rep in range(1, args.repeats + 1):
                run_id = f"{experiment_id}-{dataset['openml_id']}-{config_id}-r{rep}"
                run_dir = RUNS / run_id
                seed = seeds[rep - 1]
                result = execute(dataset, config_id, run_dir, timeout=args.timeout, seed=seed)

                scored = (
                    score(result["stdout"], dataset["cases"])
                    if result["status"] == "completed"
                    else None
                )
                write_scoring(run_dir, dataset, result, scored)

                append_result(
                    {
                        "run_id": run_id,
                        "experiment_id": experiment_id,
                        "protocol_version": PROTOCOL_VERSION,
                        "dataset_id": dataset["dataset_id"],
                        "case_id": ";".join(c["case_id"] for c in dataset["cases"]),
                        "configuration_id": config_id,
                        "repeat_index": rep,
                        "seed": seed,
                        "started_at_utc": result["started"],
                        "mlcompass_commit": _git("rev-parse", "HEAD"),
                        "provider": args.provider
                        if CONFIGURATIONS[config_id]["uses_llm"]
                        else "none",
                        "model": args.model if CONFIGURATIONS[config_id]["uses_llm"] else "none",
                        "status": result["status"],
                        "known_issues": scored["known_issues"] if scored else len(dataset["cases"]),
                        "correct_detections": scored["correct_detections"] if scored else "",
                        "missed_issues": scored["missed_issues"] if scored else "",
                        "false_positives": "",
                        "unverified_findings": scored["unverified_findings"] if scored else "",
                        "hallucinations": "",
                        "runtime_seconds": round(result["runtime"], 3),
                        "input_tokens": 0 if not CONFIGURATIONS[config_id]["uses_llm"] else "",
                        "output_tokens": 0 if not CONFIGURATIONS[config_id]["uses_llm"] else "",
                        "total_tokens": 0 if not CONFIGURATIONS[config_id]["uses_llm"] else "",
                        "estimated_llm_cost_usd": 0
                        if not CONFIGURATIONS[config_id]["uses_llm"]
                        else "",
                        "artifacts_path": f"runs/{run_id}",
                        "notes": "false_positives/hallucinations need a reviewer; see scoring.md",
                    }
                )

                mark = "ok " if result["status"] == "completed" else result["status"]
                detail = (
                    f"{scored['correct_detections']}/{scored['known_issues']} detected"
                    f", {scored['unverified_findings']} unverified"
                    if scored
                    else "not scored"
                )
                print(f"  {mark}  {run_id:<46} {result['runtime']:>6.2f}s  {detail}")
                total += 1

    print(
        f"\n{total} run(s). Results appended to {RESULTS.relative_to(BENCH)};"
        f" evidence under {RUNS.relative_to(BENCH)}/."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
