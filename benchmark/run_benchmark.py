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
  with blank detection counts, never discarded or silently retried. For an
  `llm:*` configuration a zero exit code is not enough: if the advisor
  returned nothing parseable, the configuration under test never ran, so the
  run is `failed` even though the CLI was happy. See `classify_status`.

Scoring is deliberately partial. `known_issues`, `correct_detections`,
`missed_issues` and `unverified_findings` are computed from frozen match
rules. `false_positives` and `hallucinations` are left blank: refuting a
reported issue takes judgement this script does not have, and section 4
says an unrefutable claim belongs in `unverified_findings` rather than
being counted as a false positive.

Scoring version 2.0 changed two things about how a run is read, both of
which move numbers that were already published:

- **Scope.** Scoring 1.0 scanned the whole terminal output, but the
  deterministic layer prints its warnings before the LLM is ever called, so
  every `llm:*` lane was credited with detections it inherited rather than
  produced. `scoring_scope` now cuts an `llm:*` run at the
  `✨ Recommended models` panel, where deterministic output ends and the
  advisor's begins. The inherited detections are still reported in
  `scoring.md`, labelled as inherited; they are simply not counted as the
  model's.
- **Claim type.** `unverified_findings` pooled two different things. A
  `⚠ Pitfalls` bullet asserts a defect and can be checked against the
  registry; a `🔧 Feature engineering` or `✨ Recommended models` entry is
  advice, which has no ground truth to match and never will. The two are
  now counted separately, and the old column is their sum.

A rescore of preserved evidence needs no provider and no re-execution:
`--rescore <experiment_id>` re-reads each run's `stdout.txt` and appends
fresh rows under a new experiment id, leaving the original rows in place.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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

# Bumped whenever the meaning of a scored column changes, so a row in
# `results.csv` can be read against the rules that produced it. 1.0 scored the
# whole output and reported a single `unverified_findings`; 2.0 scopes `llm:*`
# runs to the advisor's own output and splits that column by claim type.
SCORING_VERSION = "2.0"

# The CLI's own failure line when the advisor's reply will not parse
# (`src/mlcompass/cli.py::_run_advisor`). It prints this and returns None, and
# the command then exits 0 — which is why the exit code cannot be trusted to
# tell an `llm:*` run that ran from one that produced no reply at all.
ADVISOR_FAILURE_MARKER = "✗ Advisor returned an invalid response"

# Where the deterministic output ends and the advisor's begins. Rendered by
# `src/mlcompass/ui/advise.py::render_recommendation`, in this order. The
# models panel is the normal boundary; the other two are the fallback for a
# reply that carried no models, so the scope never silently swallows the
# deterministic warnings.
LLM_SECTION_MARKERS: tuple[str, ...] = (
    "✨ Recommended models",
    "🔧 Feature engineering",
    "⚠ Pitfalls",
)

# A bullet asserts a defect, or it offers advice. Only the first kind has
# anything in the registry to match against.
SECTION_HEADERS: dict[str, str] = {
    "⚠ Warnings": "warnings",
    "✨ Recommended models": "models",
    "🔧 Feature engineering": "features",
    "⚠ Pitfalls": "pitfalls",
}
SUGGESTION_SECTIONS = frozenset({"models", "features"})

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

# The model panel. A model is a *configuration*, not a footnote to one: the
# same command against a different model is a different cell, so each panel
# member produces its own `configuration_id` (`llm:<panel_id>`) rather than
# being folded into a single `llm` row whose provider column happens to
# differ. That keeps `results.csv` honest when the panel grows.
#
# Every member speaks OpenAI-compatible chat completions, which is why one
# provider value covers all four. `key_name` is looked up in `api_keys.txt`;
# a member with none needs no credential, which is the point of having a
# local lane at all.
PANEL: dict[str, dict[str, Any]] = {
    "ollama-qwen2.5-7b": {
        "provider": "openai",
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:7b",
        "key_name": None,
        "note": "local, free, no credential",
    },
    "deepseek-flash": {
        "provider": "openai",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-flash",
        "key_name": "deepseek_api",
        "note": "the published battery's provider family",
    },
    "openai-gpt-5.4-mini": {
        "provider": "openai",
        "base_url": None,
        "model": "gpt-5.4-mini",
        "key_name": "chatgpt_api",
        "note": "the pre-registered strict-capable closed lane",
    },
    "mistral-ministral-8b": {
        "provider": "openai",
        "base_url": "https://api.mistral.ai/v1",
        "model": "ministral-8b-latest",
        "key_name": "mistral_api",
        "note": "fourth provider, small hosted model",
    },
}

KEY_FILE = BENCH.parent / "api_keys.txt"


def load_keys() -> dict[str, str]:
    """Read credentials from the gitignored key file. Values are never logged."""
    keys: dict[str, str] = {}
    if KEY_FILE.exists():
        for line in KEY_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                name, _, value = line.partition("=")
                keys[name.strip()] = value.strip().strip("\"'")
    return keys


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
    "unverified_detection_claims",
    "unverified_suggestions",
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


def normalized_sha256(path: Path) -> str:
    """sha256 of the file's content with line endings normalised to LF.

    Hashing raw bytes pins the checkout's line endings, not the data. These
    CSVs are text, git is free to hand a Windows checkout CRLF and a Linux
    one LF, and the two hash differently — so a registry built on Windows
    rejects every dataset on Linux while the rows are identical. Normalising
    first makes the check say what it means: the *content* is the pinned
    content. A real change to a value still changes the digest.

    `.gitattributes` pins these files to LF in the working tree as well, so
    the two mechanisms agree rather than one papering over the other.
    """
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _verify_inputs(gt: dict[str, Any]) -> list[str]:
    """Re-hash every pinned dataset. Returns a list of problems."""
    problems = []
    for d in gt["datasets"]:
        path = BENCH / d["csv_path"]
        if not path.exists():
            problems.append(f"{d['dataset_id']}: missing {d['csv_path']}")
            continue
        actual = normalized_sha256(path)
        if actual != d["sha256"]:
            problems.append(
                f"{d['dataset_id']}: sha256 mismatch\n"
                f"    registry {d['sha256']}\n    on disk  {actual}"
            )
    return problems


# --------------------------------------------------------------------------- #
# Scoring                                                                     #
# --------------------------------------------------------------------------- #


def advisor_failed(output: str) -> bool:
    """True when the CLI reported that the advisor's reply would not parse.

    Read from the CLI's own failure line rather than inferred from the exit
    code, which is 0 in this case: `_run_advisor` catches the parse error,
    prints, and returns None, and `advise` carries on to exit cleanly.
    """
    return ADVISOR_FAILURE_MARKER in output


def classify_status(
    config_id: str, *, exit_code: int | None, stdout: str, timed_out: bool = False
) -> tuple[str, str]:
    """Return ``(status, reason)`` for one run, per protocol.md section 3.6.

    The reason is empty for a completed run and goes into `notes` otherwise.
    A configuration is only `completed` when the thing being measured
    actually ran: for an `llm:*` lane that means the advisor produced a reply
    the CLI could parse. A run whose advisor returned nothing exercised the
    deterministic layer and nothing else, so scoring it as a clean report
    would record a zero that means "no reply" as if it meant "nothing to
    report" — which is exactly what section 4's last line forbids.
    """
    if timed_out:
        return "timeout", "run exceeded the frozen timeout"
    if exit_code != 0:
        return "failed", f"CLI exited {exit_code}"
    base_id, _ = split_config(config_id)
    if CONFIGURATIONS[base_id]["uses_llm"] and advisor_failed(stdout):
        return "failed", (
            "advisor returned no parseable response, so the configuration under test"
            " was not exercised; the CLI still exited 0. Detection and finding counts"
            " left blank per protocol.md section 4."
        )
    return "completed", ""


def scoring_scope(output: str, config_id: str) -> str:
    """Narrow a run's output to the part the configuration under test wrote.

    `deterministic` owns everything it printed, so its scope is the whole
    output. An `llm:*` run's output is two halves: the deterministic analysis
    and warnings, printed before the advisor is called, and then the
    advisor's own panels. Scoring the whole thing credits the model with
    findings the rule-based layer handed it, which is how every LLM lane came
    to read 9/9 on a measure that was supposed to distinguish them.
    """
    base_id, _ = split_config(config_id)
    if not CONFIGURATIONS[base_id]["uses_llm"]:
        return output
    cuts = [output.index(m) for m in LLM_SECTION_MARKERS if m in output]
    return output[min(cuts) :] if cuts else ""


def split_findings(output: str) -> list[tuple[str, str]]:
    """Pull the bulleted findings out of a rendered report, with their section.

    `protocol.md` section 4 asks for compound reports to be split into
    distinct claims and deduplicated. mlcompass renders one warning per
    bullet, so the bullet is the claim boundary; rich wraps long lines, so
    a continuation line (no bullet marker, indented) folds into the claim
    above it rather than becoming a claim of its own.

    The section a bullet sits under is carried alongside it because it is
    what decides the claim's *kind*, and the kind decides what the claim can
    be scored against. A bullet before any recognised header keeps the
    section `""`, which is treated as a detection claim — an unattributed
    assertion is read strictly rather than excused as advice.
    """
    findings: list[tuple[str, str]] = []
    section = ""
    for raw in output.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        header = next((v for k, v in SECTION_HEADERS.items() if stripped.startswith(k)), None)
        if header is not None:
            section = header
            continue
        if stripped.startswith(("•", "-", "*")) and len(stripped) > 2:
            findings.append((section, stripped.lstrip("•-* ").strip()))
        elif findings and line.startswith(" ") and stripped:
            sec, text = findings[-1]
            findings[-1] = (sec, text + " " + stripped)
    # Deduplicate while preserving order; the first occurrence keeps its section.
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for sec, text in findings:
        key = re.sub(r"\s+", " ", text.lower())
        if key not in seen:
            seen.add(key)
            out.append((sec, text))
    return out


def _split_findings(output: str) -> list[str]:
    """Bullet texts alone, for callers that do not care which section they came from."""
    return [text for _, text in split_findings(output)]


def _match(cases: list[dict[str, Any]], text: str) -> list[str]:
    """Issue ids whose pattern and required values both occur in ``text``."""
    haystack = re.sub(r"\s+", " ", text.lower())
    hits = []
    for case in cases:
        if not re.search(case["match_pattern"].lower(), haystack):
            continue
        if not all(str(v).lower() in haystack for v in case.get("required_values", [])):
            continue
        hits.append(case["issue_id"])
    return hits


def score(
    output: str, cases: list[dict[str, Any]], *, config_id: str = "deterministic"
) -> dict[str, Any]:
    """Match a run's output against the frozen cases for its dataset.

    Scored over the configuration's own output only (see `scoring_scope`).
    Detections the deterministic layer contributed to an `llm:*` run are
    returned under `inherited_detected_ids` so they stay visible in
    `scoring.md` without being counted as the model's.
    """
    scoped = scoring_scope(output, config_id)
    inherited_region = output[: len(output) - len(scoped)] if scoped else output
    findings = split_findings(scoped)
    haystack = re.sub(r"\s+", " ", scoped.lower())

    detected: list[str] = []
    matched_findings: set[int] = set()
    for case in cases:
        pattern = case["match_pattern"].lower()
        if not re.search(pattern, haystack):
            continue
        if not all(str(v).lower() in haystack for v in case.get("required_values", [])):
            continue
        detected.append(case["issue_id"])
        for i, (_, f) in enumerate(findings):
            flat = re.sub(r"\s+", " ", f.lower())
            if re.search(pattern, flat):
                matched_findings.add(i)

    unverified = [(s, f) for i, (s, f) in enumerate(findings) if i not in matched_findings]
    suggestions = [(s, f) for s, f in unverified if s in SUGGESTION_SECTIONS]
    detection_claims = [(s, f) for s, f in unverified if s not in SUGGESTION_SECTIONS]
    base_id, _ = split_config(config_id)
    scoped_to_llm = CONFIGURATIONS[base_id]["uses_llm"]
    return {
        "known_issues": len(cases),
        "correct_detections": len(detected),
        "missed_issues": len(cases) - len(detected),
        "detected_ids": detected,
        "missed_ids": [c["issue_id"] for c in cases if c["issue_id"] not in detected],
        "unverified_findings": len(unverified),
        "unverified_detection_claims": len(detection_claims),
        "unverified_suggestions": len(suggestions),
        "unverified_texts": [f for _, f in unverified],
        "detection_claim_texts": [f for _, f in detection_claims],
        "suggestion_texts": [f for _, f in suggestions],
        "all_findings": [f for _, f in findings],
        "inherited_detected_ids": _match(cases, inherited_region) if scoped_to_llm else [],
        "inherited_findings": [f for _, f in split_findings(inherited_region)]
        if scoped_to_llm
        else [],
        "scope": (
            f"LLM output only, from the first of {', '.join(LLM_SECTION_MARKERS)} onward"
            if scoped_to_llm
            else "full output"
        ),
        "scoring_version": SCORING_VERSION,
    }


# --------------------------------------------------------------------------- #
# Execution                                                                   #
# --------------------------------------------------------------------------- #


def split_config(config_id: str) -> tuple[str, dict[str, Any] | None]:
    """Split ``llm:<panel_id>`` into its configuration and panel member."""
    base, _, panel_id = config_id.partition(":")
    if not panel_id:
        return base, None
    if panel_id not in PANEL:
        raise KeyError(f"unknown panel member {panel_id!r}; known: {sorted(PANEL)}")
    return base, PANEL[panel_id]


def _describe_config(config_id: str) -> str:
    """One plan line per configuration, naming the model when there is one."""
    base_id, member = split_config(config_id)
    line = f"- `{config_id}`: {CONFIGURATIONS[base_id]['description']}"
    if member is not None:
        endpoint = member["base_url"] or "the provider's default endpoint"
        line += (
            f" Model `{member['model']}` via `{member['provider']}` at {endpoint}"
            f" — {member['note']}."
        )
    return line


def execute(
    dataset: dict[str, Any],
    config_id: str,
    run_dir: Path,
    *,
    timeout: int,
    seed: int,
) -> dict[str, Any]:
    """Run one cell in a fresh workspace and preserve its evidence."""
    base_id, member = split_config(config_id)
    config = CONFIGURATIONS[base_id]
    csv_path = (BENCH / dataset["csv_path"]).resolve()
    argv = [a.format(csv=str(csv_path), target=dataset["target"]) for a in config["argv"]]

    env = dict(os.environ)
    if member is not None:
        argv += ["--provider", member["provider"], "--model", member["model"]]
        if member["base_url"]:
            argv += ["--base-url", member["base_url"]]
        # The CLI reads the credential from the provider's own variable, so
        # the key is placed in the child's environment rather than on the
        # command line — an argv ends up in `command.json`, and a run record
        # that leaks a key is worse than no record.
        key = load_keys().get(member["key_name"], "") if member["key_name"] else "local"
        env["OPENAI_API_KEY"] = key or "local"

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
            env=env,
        )
        runtime = time.monotonic() - clock
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
        status, reason = classify_status(config_id, exit_code=code, stdout=stdout)
    except subprocess.TimeoutExpired as exc:
        runtime = time.monotonic() - clock
        code = None
        stdout = exc.stdout.decode("utf-8", "replace") if exc.stdout else ""
        stderr = exc.stderr.decode("utf-8", "replace") if exc.stderr else ""
        status, reason = classify_status(config_id, exit_code=code, stdout=stdout, timed_out=True)
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
                "status_reason": reason,
                "runtime_seconds": round(runtime, 3),
                "seed": seed,
                "input_sha256": normalized_sha256(csv_path),
                "input_sha256_basis": "content with CRLF and CR normalised to LF",
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
        "reason": reason,
        "stdout": stdout,
        "stderr": stderr,
        "runtime": runtime,
        "started": started,
    }


def write_scoring(
    run_dir: Path,
    dataset: dict[str, Any],
    result: dict[str, Any],
    scored: dict[str, Any] | None,
    *,
    config_id: str = "deterministic",
    registry_version: str = "",
) -> None:
    lines = [
        f"# Scoring — {run_dir.name}",
        "",
        f"Dataset: {dataset['dataset_id']} ({dataset['name']})",
        f"Configuration: {config_id}",
        f"Status: {result['status']}",
        f"Scoring version: {SCORING_VERSION}"
        + (f"; ground-truth registry {registry_version}" if registry_version else ""),
        "",
    ]
    if result.get("reason"):
        lines += [f"Reason: {result['reason']}", ""]
    if scored is None:
        lines += [
            "Run did not complete; detection and finding counts are left blank per",
            "protocol.md section 4, which says not to treat a missing output as a",
            "successful clean report. The run is kept, not discarded (section 3.6).",
            "",
        ]
    else:
        lines += [
            f"Scope scored: {scored['scope']}.",
            "",
            f"known_issues: {scored['known_issues']}",
            f"correct_detections: {scored['correct_detections']}  {scored['detected_ids']}",
            f"missed_issues: {scored['missed_issues']}  {scored['missed_ids']}",
            f"unverified_findings: {scored['unverified_findings']}"
            f"  (= {scored['unverified_detection_claims']} unmatched detection claims"
            f" + {scored['unverified_suggestions']} suggestions)",
            "",
            "A detection claim asserts a defect and is checked against the registry;",
            "it comes from `⚠ Warnings` or `⚠ Pitfalls`. A suggestion comes from",
            "`🔧 Feature engineering` or `✨ Recommended models` and has no ground",
            "truth to match against, so it is counted apart rather than pooled with",
            "claims that could have matched and did not.",
            "",
            "## Findings in scope",
            "",
        ]
        lines += [f"- {f}" for f in scored["all_findings"]] or ["- (none)"]
        lines += ["", "## Unmatched detection claims (unverified, not refuted)", ""]
        lines += [f"- {f}" for f in scored["detection_claim_texts"]] or ["- (none)"]
        lines += ["", "## Suggestions (no ground truth to match against)", ""]
        lines += [f"- {f}" for f in scored["suggestion_texts"]] or ["- (none)"]
        if scored["scope"] != "full output":
            lines += [
                "",
                "## Inherited from the deterministic layer — context, not counted",
                "",
                "These were printed before the advisor was called, so they are not",
                "this configuration's output. They are shown so nothing is hidden.",
                "",
                f"Detections inherited: {scored['inherited_detected_ids'] or '(none)'}",
                "",
            ]
            lines += [f"- {f}" for f in scored["inherited_findings"]] or ["- (none)"]
        lines += [
            "",
            "false_positives and hallucinations are blank: refuting a reported",
            "issue needs a reviewer, and protocol.md section 4 puts an unrefutable",
            "claim in unverified_findings rather than in false_positives.",
        ]
    (run_dir / "scoring.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _notes(registry_version: str, reason: str = "", extra: str = "") -> str:
    """The `notes` cell: which rules produced this row, then anything unusual.

    Every row says the registry and scoring version it was produced under, so
    a reader comparing two rows can tell a change in the tool from a change in
    how it was read. Semicolons, not commas: this lands in a CSV.
    """
    parts = [f"registry {registry_version}; scoring {SCORING_VERSION}"]
    if extra:
        parts.append(extra)
    if reason:
        parts.append(reason)
    parts.append("false_positives/hallucinations need a reviewer; see scoring.md")
    return " | ".join(parts)


def _migrate_results_header() -> None:
    """Widen `results.csv` to the current columns, keeping every existing value.

    Scoring 2.0 adds two columns. A row written under 1.0 has no value for
    them and must not be given one — the old scoring did not compute the
    split, and inventing a number for it would be worse than a blank. So the
    header is rewritten and the old rows are padded with empty cells, which
    is what "this was not measured" looks like in a CSV.
    """
    if not (RESULTS.exists() and RESULTS.read_text(encoding="utf-8").strip()):
        return
    with RESULTS.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames == RESULT_COLUMNS:
            return
        rows = list(reader)
    with RESULTS.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (row.get(k) or "") for k in RESULT_COLUMNS})


def append_result(row: dict[str, Any]) -> None:
    _migrate_results_header()
    exists = RESULTS.exists() and RESULTS.read_text(encoding="utf-8").strip()
    with RESULTS.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULT_COLUMNS})


# --------------------------------------------------------------------------- #
# Rescoring preserved evidence                                                #
# --------------------------------------------------------------------------- #


def _known_config_ids() -> list[str]:
    return ["deterministic"] + [f"llm:{p}" for p in sorted(PANEL)]


def _identify_run(run_name: str, experiment_id: str, gt: dict[str, Any]) -> tuple[Any, str, int]:
    """Recover ``(dataset, config_id, repeat)`` from a run directory name.

    Run ids are built as ``<experiment>-<openml_id>-<config>-r<n>`` with the
    colon in ``llm:<panel>`` flattened to a dash, so the mapping back is a
    match against the known ids rather than a guess at where the dashes fall.
    """
    for dataset in gt["datasets"]:
        for config_id in _known_config_ids():
            prefix = f"{experiment_id}-{dataset['openml_id']}-{config_id.replace(':', '-')}-r"
            if run_name.startswith(prefix) and run_name[len(prefix) :].isdigit():
                return dataset, config_id, int(run_name[len(prefix) :])
    raise KeyError(run_name)


def experiment_already_recorded(experiment_id: str) -> bool:
    """True when `results.csv` already carries rows for that experiment id."""
    if not (RESULTS.exists() and RESULTS.read_text(encoding="utf-8").strip()):
        return False
    with RESULTS.open(newline="", encoding="utf-8") as fh:
        return any(row.get("experiment_id") == experiment_id for row in csv.DictReader(fh))


def rescore(experiment_id: str, gt: dict[str, Any], new_experiment_id: str) -> int:
    """Re-score an experiment's preserved output without re-executing anything.

    The evidence is on disk — that is the point of `protocol.md` section 3.5
    keeping it. Original rows stay exactly where they are and the new rows
    arrive under `new_experiment_id`, so the correction is visible as a
    correction rather than as a silently better number. The original
    `scoring.md` of each run is kept alongside the regenerated one, per
    section 4's requirement to preserve the original annotations.
    """
    if experiment_already_recorded(new_experiment_id):
        raise ValueError(
            f"{new_experiment_id} already has rows in {RESULTS.name}; rescoring into it"
            " again would duplicate every one of them. Pass --rescore-id with a"
            " different id, or remove those rows deliberately first."
        )
    run_dirs = sorted(d for d in RUNS.glob(f"{experiment_id}-*") if d.is_dir())
    identified = []
    for run_dir in run_dirs:
        try:
            dataset, config_id, rep = _identify_run(run_dir.name, experiment_id, gt)
        except KeyError:
            print(f"  skip     {run_dir.name}: not a run of a known configuration")
            continue
        if not (run_dir / "stdout.txt").exists() or not (run_dir / "command.json").exists():
            print(f"  skip     {run_dir.name}: no preserved stdout/command record")
            continue
        dataset_order = [d["dataset_id"] for d in gt["datasets"]].index(dataset["dataset_id"])
        identified.append(((dataset_order, config_id, rep), run_dir, dataset, config_id, rep))

    total = 0
    for _, run_dir, dataset, config_id, rep in sorted(identified, key=lambda x: x[0]):
        stdout = (run_dir / "stdout.txt").read_text(encoding="utf-8")
        cmd = json.loads((run_dir / "command.json").read_text(encoding="utf-8"))
        _, member = split_config(config_id)

        status, reason = classify_status(
            config_id,
            exit_code=cmd.get("exit_code"),
            stdout=stdout,
            timed_out=cmd.get("status") == "timeout",
        )
        scored = (
            score(stdout, dataset["cases"], config_id=config_id) if status == "completed" else None
        )

        # Keep the annotation the old scoring produced before overwriting it.
        old = run_dir / "scoring.md"
        kept = run_dir / "scoring.scoring-1.0.md"
        if old.exists() and not kept.exists():
            kept.write_text(old.read_text(encoding="utf-8"), encoding="utf-8")

        result = {"status": status, "reason": reason}
        write_scoring(
            run_dir,
            dataset,
            result,
            scored,
            config_id=config_id,
            registry_version=gt["registry_version"],
        )

        append_result(
            {
                "run_id": run_dir.name,
                "experiment_id": new_experiment_id,
                "protocol_version": PROTOCOL_VERSION,
                "dataset_id": dataset["dataset_id"],
                "case_id": ";".join(c["case_id"] for c in dataset["cases"]),
                "configuration_id": config_id,
                "repeat_index": rep,
                "seed": cmd.get("seed", ""),
                "started_at_utc": cmd.get("started_at_utc", ""),
                "mlcompass_commit": cmd.get("mlcompass_commit", ""),
                "provider": member["provider"] if member else "none",
                "model": member["model"] if member else "none",
                "status": status,
                "known_issues": len(dataset["cases"]),
                "correct_detections": scored["correct_detections"] if scored else "",
                "missed_issues": scored["missed_issues"] if scored else "",
                "false_positives": "",
                "unverified_findings": scored["unverified_findings"] if scored else "",
                "unverified_detection_claims": (
                    scored["unverified_detection_claims"] if scored else ""
                ),
                "unverified_suggestions": scored["unverified_suggestions"] if scored else "",
                "hallucinations": "",
                "runtime_seconds": cmd.get("runtime_seconds", ""),
                "input_tokens": 0 if member is None else "",
                "output_tokens": 0 if member is None else "",
                "total_tokens": 0 if member is None else "",
                "estimated_llm_cost_usd": 0 if member is None else "",
                "artifacts_path": f"runs/{run_dir.name}",
                "notes": _notes(
                    gt["registry_version"],
                    reason,
                    extra=(
                        f"rescored from the preserved stdout of {experiment_id}; not re-executed"
                    ),
                ),
            }
        )

        mark = "ok " if status == "completed" else status
        detail = (
            f"{scored['correct_detections']}/{scored['known_issues']} detected"
            f", {scored['unverified_detection_claims']} unmatched claims"
            f", {scored['unverified_suggestions']} suggestions"
            if scored
            else "not scored"
        )
        print(f"  {mark}  {run_dir.name:<46} {detail}")
        total += 1
    return total


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
                *[_describe_config(c) for c in config_ids],
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
        help=(
            "configuration to run, repeatable; default deterministic. "
            "A model lane is llm:<panel_id>, e.g. llm:ollama-qwen2.5-7b. "
            f"Configurations: {', '.join(sorted(CONFIGURATIONS))}. "
            f"Panel: {', '.join(sorted(PANEL))}"
        ),
    )
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--seed", type=int, action="append", help="seed per repetition (repeatable)")
    ap.add_argument("--plan", action="store_true", help="freeze a plan and exit without running")
    ap.add_argument("--dry-run", action="store_true", help="print the cells and exit")
    ap.add_argument("--panel", action="store_true", help="expand to every panel member as llm:<id>")
    ap.add_argument(
        "--rescore",
        metavar="EXPERIMENT_ID",
        help=(
            "re-score that experiment's preserved stdout under the current registry "
            "and scoring version, appending rows under a new experiment id. "
            "Executes nothing and leaves the original rows in place."
        ),
    )
    ap.add_argument(
        "--rescore-id",
        default=None,
        help="experiment id for the rescored rows; default <EXPERIMENT_ID>-rescored",
    )
    args = ap.parse_args()

    gt = _load_ground_truth()
    config_ids = list(args.config or ["deterministic"])
    if args.panel:
        config_ids = [c for c in config_ids if c != "llm"] + [f"llm:{p}" for p in sorted(PANEL)]
    seeds = args.seed or list(range(args.repeats))
    if len(seeds) < args.repeats:
        seeds = (seeds * args.repeats)[: args.repeats]

    if args.rescore:
        new_id = args.rescore_id or f"{args.rescore}-rescored"
        print(
            f"Rescoring {args.rescore} from preserved evidence under registry"
            f" {gt['registry_version']}, scoring {SCORING_VERSION}.\n"
            f"Nothing is executed; new rows are appended as {new_id}.\n"
        )
        try:
            count = rescore(args.rescore, gt, new_id)
        except ValueError as exc:
            print(f"Refusing to rescore: {exc}")
            return 1
        print(f"\n{count} run(s) rescored. Original rows for {args.rescore} are unchanged.")
        return 0

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

    total = 0
    for dataset in gt["datasets"]:
        for config_id in config_ids:
            for rep in range(1, args.repeats + 1):
                _, member = split_config(config_id)
                safe_cfg = config_id.replace(":", "-")
                run_id = f"{experiment_id}-{dataset['openml_id']}-{safe_cfg}-r{rep}"
                run_dir = RUNS / run_id
                seed = seeds[rep - 1]
                result = execute(dataset, config_id, run_dir, timeout=args.timeout, seed=seed)

                scored = (
                    score(result["stdout"], dataset["cases"], config_id=config_id)
                    if result["status"] == "completed"
                    else None
                )
                write_scoring(
                    run_dir,
                    dataset,
                    result,
                    scored,
                    config_id=config_id,
                    registry_version=gt["registry_version"],
                )

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
                        "provider": member["provider"] if member else "none",
                        "model": member["model"] if member else "none",
                        "status": result["status"],
                        "known_issues": scored["known_issues"] if scored else len(dataset["cases"]),
                        "correct_detections": scored["correct_detections"] if scored else "",
                        "missed_issues": scored["missed_issues"] if scored else "",
                        "false_positives": "",
                        "unverified_findings": scored["unverified_findings"] if scored else "",
                        "unverified_detection_claims": (
                            scored["unverified_detection_claims"] if scored else ""
                        ),
                        "unverified_suggestions": (
                            scored["unverified_suggestions"] if scored else ""
                        ),
                        "hallucinations": "",
                        "runtime_seconds": round(result["runtime"], 3),
                        "input_tokens": 0 if member is None else "",
                        "output_tokens": 0 if member is None else "",
                        "total_tokens": 0 if member is None else "",
                        "estimated_llm_cost_usd": 0 if member is None else "",
                        "artifacts_path": f"runs/{run_id}",
                        "notes": _notes(gt["registry_version"], result.get("reason", "")),
                    }
                )

                mark = "ok " if result["status"] == "completed" else result["status"]
                detail = (
                    f"{scored['correct_detections']}/{scored['known_issues']} detected"
                    f", {scored['unverified_detection_claims']} unmatched claims"
                    f", {scored['unverified_suggestions']} suggestions"
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
