"""Command-line interface for ``mlcompass``.

Subcommands are added incrementally as the project advances through
its phases. See ARCHITECTURE.md §7 for the full CLI design.

Currently implemented:
    init      — create a new ``.mlcompass/`` project (Faz 1)
    advise    — analyze dataset + recommend models / features / pitfalls (Faz 1)
    audit     — static analysis of a training script (Faz 2a)
    watch     — monitor a training log for anomalies (Faz 2b)
    compare   — diff two training runs side-by-side (Faz 2c)
    evaluate  — post-training analysis on a predictions table (Faz 3a)
    deploy    — deployment readiness check (Faz 4a)
    status    — summarise the active project context (Faz 5)
    agent     — self-driving agent over the eight mlcompass tools (Faz 7)
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel

from . import __version__
from .agents._common import (
    DEFAULT_PROVIDER,
    ENV_BASE_URL,
    ENV_MODEL,
    ENV_PROVIDER,
    PROVIDER_ANTHROPIC,
    PROVIDER_KEY_ENV,
    SUPPORTED_PROVIDERS,
    build_client,
    resolve_llm_config,
)
from .agents.advise import AdvisorParseError, get_recommendation
from .agents.audit import AuditAgentError, prioritize_findings
from .agents.compare import CompareAgentError, hypothesize_comparison
from .agents.deploy import DeployAgentError, advise_deployment
from .agents.evaluate import EvaluateAgentError, interpret_evaluation
from .agents.leakage_investigator import (
    LEAKAGE_MODEL_DEFAULT,
    LeakageAgentError,
    investigate_leakage_bound,
)
from .agents.monitor import MonitorAgentError, interpret_drift
from .agents.optimize import OptimizeAgentError, strategize_optimize
from .agents.watch import WatchAgentError, diagnose_findings
from .context import ProjectContext, ProjectExistsError, ProjectNotFoundError
from .tools.anomaly import run_all_detectors
from .tools.config_edit import (
    ApplyResult,
    ConfigEditError,
    apply_edits,
)
from .tools.dataset import analyze_dataset
from .tools.deploy import (
    SUPPORTED_TARGETS as DEPLOY_TARGETS,
)
from .tools.deploy import (
    DeployAnalysisError,
    assess_deployment,
)
from .tools.drift import DriftAnalysisError, detect_drift
from .tools.drift import load_table as load_drift_table
from .tools.evaluation import (
    EvaluationError,
    load_results,
)
from .tools.evaluation import (
    evaluate as run_evaluation,
)
from .tools.logs import (
    load_snapshots,
    merge_consecutive_same_epoch,
)
from .tools.optimize import (
    OptimizeAnalysisError,
    load_runs_from_dir,
    optimize_history,
    parse_constraints,
)
from .tools.runs import RunNotFoundError, compare_runs, load_run
from .tools.script import audit_script
from .ui.advise import render_analysis, render_recommendation
from .ui.audit import render_audit, render_audit_priorities
from .ui.compare import render_compare, render_compare_hypothesis
from .ui.config_edit import make_console_confirm, render_apply_summary
from .ui.deploy import render_deployment, render_deployment_advice
from .ui.evaluate import (
    render_evaluation,
    render_evaluation_interpretation,
    render_leakage_narration,
)
from .ui.monitor import render_drift, render_drift_interpretation
from .ui.optimize import render_optimize, render_optimize_strategy
from .ui.status import render_status
from .ui.watch import render_new_findings, render_watch_diagnosis, render_watch_report


def _force_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8.

    On Windows the default encoding follows the system code page
    (e.g. ``cp1254`` for Turkish locales) and cannot represent the
    Unicode glyphs rich uses for panels, checkmarks, and box drawing.
    Reconfiguring early avoids ``UnicodeEncodeError`` at print time.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(AttributeError, ValueError):
            reconfigure(encoding="utf-8", errors="replace")


_force_utf8_stdio()
console = Console()


# --------------------------------------------------------------------------- #
# LLM provider selection, shared by every command that has --llm              #
# --------------------------------------------------------------------------- #


def _llm_target(
    provider: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str | None]:
    """Resolve which provider and endpoint ``--llm`` would use.

    Same precedence as the agents themselves: the explicit flag, then
    ``MLCOMPASS_LLM_PROVIDER`` / ``MLCOMPASS_LLM_BASE_URL``, then the
    built-in default (Anthropic, and the SDK's own endpoint).
    """
    resolved_provider = (provider or os.environ.get(ENV_PROVIDER) or DEFAULT_PROVIDER).strip()
    resolved_base_url = base_url or os.environ.get(ENV_BASE_URL) or None
    return resolved_provider, resolved_base_url


def _has_api_key(
    provider: str | None = None,
    base_url: str | None = None,
) -> bool:
    """True iff ``--llm`` has a provider it can actually reach.

    A credential in the provider's own variable, or a base URL — a local
    ollama / vLLM server authenticates nobody, so requiring a key there
    would refuse the one configuration that costs nothing.
    """
    resolved_provider, resolved_base_url = _llm_target(provider, base_url)
    if resolved_provider not in PROVIDER_KEY_ENV:
        return False
    if resolved_base_url:
        return True
    return bool(os.environ.get(PROVIDER_KEY_ENV[resolved_provider]))


def _llm_unavailable_reason(
    provider: str | None = None,
    base_url: str | None = None,
) -> str:
    """Explain, in one clause, why ``--llm`` cannot run."""
    resolved_provider, _ = _llm_target(provider, base_url)
    if resolved_provider not in PROVIDER_KEY_ENV:
        return (
            f"unknown provider {resolved_provider!r} (supported: {', '.join(SUPPORTED_PROVIDERS)})"
        )
    key_env = PROVIDER_KEY_ENV[resolved_provider]
    if resolved_provider == PROVIDER_ANTHROPIC:
        return f"{key_env} is unset"
    return f"{key_env} is unset and no --base-url was given"


def _warn_llm_unavailable(
    role: str,
    provider: str | None = None,
    base_url: str | None = None,
) -> None:
    """Print the standard skip warning for an unreachable provider."""
    console.print(
        f"\n[yellow]⚠ --llm requested but {_llm_unavailable_reason(provider, base_url)}; "
        f"skipping {role}.[/yellow]\n"
        "[dim]  Set that variable, or run a local model for free: "
        "--provider openai --base-url http://localhost:11434/v1 --model qwen2.5:7b[/dim]"
    )


def _llm_available(
    role: str,
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> bool:
    """True when ``--llm`` can run; otherwise print why and return False.

    Two ways to be unrunnable: nothing to authenticate or connect to, and —
    for a non-Anthropic provider — no model, because the built-in default is
    a Claude model name and sending it elsewhere would be nonsense.
    """
    if not _has_api_key(provider, base_url):
        _warn_llm_unavailable(role, provider, base_url)
        return False

    resolved_provider, _ = _llm_target(provider, base_url)
    if resolved_provider != PROVIDER_ANTHROPIC and not (model or os.environ.get(ENV_MODEL)):
        console.print(
            f"\n[yellow]⚠ --llm with --provider {resolved_provider} needs --model "
            f"(or ${ENV_MODEL}); skipping {role}.[/yellow]"
        )
        return False
    return True


def llm_target_options(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Add ``--provider`` / ``--base-url`` to a command that has ``--llm``.

    Every ``--llm`` command accepts the same pair, so they are declared
    once. Both default to None so that an unset flag falls through to
    ``MLCOMPASS_LLM_PROVIDER`` / ``MLCOMPASS_LLM_BASE_URL`` and then to the
    built-in default, rather than silently overriding the environment with
    a click default.
    """
    fn = click.option(
        "--base-url",
        "llm_base_url",
        default=None,
        metavar="URL",
        help=(
            "Endpoint for --provider openai, e.g. http://localhost:11434/v1 "
            "for a local ollama server (no API key needed). "
            f"Falls back to ${ENV_BASE_URL}."
        ),
    )(fn)
    fn = click.option(
        "--provider",
        "llm_provider",
        type=click.Choice(list(SUPPORTED_PROVIDERS)),
        default=None,
        help=(
            f"LLM provider for --llm (default: {DEFAULT_PROVIDER}). 'openai' means "
            "any OpenAI-compatible Chat Completions endpoint. "
            f"Falls back to ${ENV_PROVIDER}."
        ),
    )(fn)
    return fn


#: Reused in every ``--model`` help string, since the resolution order is
#: the same everywhere.
MODEL_HELP_SUFFIX = (
    "Defaults to the built-in Claude model, or $MLCOMPASS_LLM_MODEL. "
    "Required with --provider openai."
)


@click.group(
    help="mlcompass — your AI ML engineer at every pipeline stage.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, prog_name="mlcompass")
def cli() -> None:
    """Root command group."""


@cli.command(help="Initialize a new mlcompass project.")
@click.argument("name")
@click.option(
    "--path",
    "parent_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Directory in which .mlcompass/ will be created.",
)
@click.option(
    "--default-model",
    default="claude-opus-4-7",
    show_default=True,
    help="Default LLM model for this project.",
)
def init(name: str, parent_dir: Path, default_model: str) -> None:
    """Create a new mlcompass project under ``parent_dir/.mlcompass/``."""
    try:
        ctx = ProjectContext.init(
            name=name,
            parent_dir=parent_dir,
            default_model=default_model,
        )
    except ProjectExistsError as e:
        console.print(f"[red]✗[/red] {e}")
        raise SystemExit(1) from e

    console.print(
        Panel.fit(
            f"[green]✓[/green] Project [bold]{name}[/bold] initialized.\n\n"
            f"Directory:  [cyan]{ctx.path}[/cyan]\n"
            f"Model:      {default_model}\n\n"
            f"Next: [yellow]mlcompass advise <data.csv>[/yellow]",
            title="mlcompass init",
            border_style="green",
        )
    )


@cli.command(help="Analyze a dataset and recommend models, features, and pitfalls.")
@click.argument(
    "dataset_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--target",
    "target_column",
    default=None,
    help="Target column name (auto-detected from name conventions if omitted).",
)
@click.option(
    "--sample-rows",
    type=int,
    default=None,
    help="Limit analysis to the first N rows (useful for very large files).",
)
@click.option(
    "--llm",
    is_flag=True,
    help="Add the LLM advisor step on top of the deterministic analysis.",
)
@click.option(
    "--no-llm",
    is_flag=True,
    help="Deprecated and now redundant: the advisor is opt-in via --llm.",
)
@llm_target_options
@click.option(
    "--model",
    "advisor_model",
    default=None,
    help=f"Model used by the advisor. {MODEL_HELP_SUFFIX}",
)
def advise(
    dataset_path: Path,
    target_column: str | None,
    sample_rows: int | None,
    llm: bool,
    no_llm: bool,
    llm_provider: str | None,
    llm_base_url: str | None,
    advisor_model: str | None,
) -> None:
    """Run the deterministic dataset analyzer, then the LLM advisor."""
    project = _try_load_project()

    with console.status("[cyan]Analyzing dataset...[/cyan]", spinner="dots"):
        analysis = analyze_dataset(
            dataset_path,
            target_column=target_column,
            sample_rows=sample_rows,
        )

    render_analysis(console, analysis)

    recommendation: dict[str, Any] | None = None

    # Opt-in, like every other LLM-capable command. It used to be the one
    # exception: the advisor ran by default, so a bare ``advise`` on a
    # machine with ANTHROPIC_API_KEY exported made a paid call without
    # being asked. ``--no-llm`` still parses so existing scripts keep
    # working, but it no longer does anything the default does not.
    if no_llm and not llm:
        console.print("\n[dim](--no-llm is redundant now; the advisor is opt-in via --llm)[/dim]")
    elif not llm:
        console.print("\n[dim](deterministic analysis only; pass --llm to add the advisor)[/dim]")
    elif not _llm_available("advisor step", llm_provider, llm_base_url, advisor_model):
        pass  # _llm_available already explained why.
    else:
        recommendation = _run_advisor(
            analysis,
            model=advisor_model,
            provider=llm_provider,
            base_url=llm_base_url,
        )
        if recommendation is not None:
            render_recommendation(console, recommendation)

    if project is not None:
        _persist_advise_result(
            project=project,
            dataset_path=dataset_path,
            analysis=analysis,
            recommendation=recommendation,
        )


# --------------------------------------------------------------------------- #
# advise helpers (split out so tests can monkeypatch them)                    #
# --------------------------------------------------------------------------- #


def _try_load_project() -> ProjectContext | None:
    """Load an existing project context, or warn and return None."""
    try:
        return ProjectContext.load()
    except ProjectNotFoundError:
        console.print(
            "[dim](no .mlcompass/ project found in current path — "
            "running standalone; results will not be persisted)[/dim]\n"
        )
        return None


# Indirection points: tests monkeypatch these to inject fakes instead of
# calling the real agents.
_advisor_callable: Callable[..., dict[str, Any]] = get_recommendation
_audit_prioritizer_callable: Callable[..., dict[str, Any]] = prioritize_findings
_watch_diagnostician_callable: Callable[..., dict[str, Any]] = diagnose_findings
_compare_hypothesizer_callable: Callable[..., dict[str, Any]] = hypothesize_comparison

# Tests can also replace the apply-edits driver to bypass real disk writes.
_apply_edits_callable: Callable[..., ApplyResult] = apply_edits
_evaluate_interpreter_callable: Callable[..., dict[str, Any]] = interpret_evaluation
_deploy_advisor_callable: Callable[..., dict[str, Any]] = advise_deployment


def _run_advisor(
    analysis: dict[str, Any],
    *,
    model: str | None,
    provider: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any] | None:
    """Invoke the LLM advisor, handling parse errors gracefully."""
    try:
        with console.status(
            "[cyan]Consulting model advisor...[/cyan]",
            spinner="dots",
        ):
            return _advisor_callable(
                analysis,
                model=model,
                provider=provider,
                base_url=base_url,
            )
    except AdvisorParseError as exc:
        console.print(f"\n[red]✗ Advisor returned an invalid response:[/red] {exc}")
        return None


def _persist_advise_result(
    *,
    project: ProjectContext,
    dataset_path: Path,
    analysis: dict[str, Any],
    recommendation: dict[str, Any] | None,
) -> None:
    """Save dataset metadata + context updates + advice log entry."""
    fingerprint = project.register_dataset(
        dataset_path,
        meta={"analysis": analysis},
    )

    project.write_context(
        {
            "active_dataset": f"datasets/{fingerprint}.json",
            "project_type": analysis["task_hint"].get("type"),
            "target_column": analysis["target_hint"].get("column"),
        }
    )

    top_model: str
    if recommendation and recommendation.get("models"):
        top_model = recommendation["models"][0].get("name", "n/a")
    else:
        top_model = "n/a (LLM advisor skipped)"

    project.append_decision(
        command="advise",
        summary=f"Top model recommendation: {top_model}",
        reasoning=(
            f"Dataset: {dataset_path}, task: {analysis['task_hint'].get('type', 'unknown')}"
        ),
    )

    _append_advice_log(project, dataset_path, analysis, recommendation)

    console.print(f"\n[green]✓[/green] Saved to {project.path}")


def _append_advice_log(
    project: ProjectContext,
    dataset_path: Path,
    analysis: dict[str, Any],
    recommendation: dict[str, Any] | None,
) -> None:
    """Append a single advice entry to ``.mlcompass/advice.log``."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "task_type": analysis["task_hint"].get("type"),
        "target": analysis["target_hint"].get("column"),
        "recommendation": recommendation,
    }
    log_path = project.path / "advice.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------- #
# audit command                                                               #
# --------------------------------------------------------------------------- #


@cli.command(help="Statically analyze a training script for common ML mistakes.")
@click.argument(
    "script_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--skip",
    "skip_rules",
    multiple=True,
    type=click.Choice(
        [
            "seed",
            "val_split",
            "optimizer",
            "loss_stability",
            "dataloader",
            "grad_clipping",
            "eval_mode",
            "batch_size",
        ]
    ),
    help="Rule IDs to skip (may be repeated).",
)
@click.option(
    "--llm",
    "use_llm",
    is_flag=True,
    help="After the static analysis, ask Claude to prioritize the findings.",
)
@llm_target_options
@click.option(
    "--model",
    "llm_model",
    default=None,
    help=f"Model used by the prioritizer when --llm is set. {MODEL_HELP_SUFFIX}",
)
def audit(
    script_path: Path,
    skip_rules: tuple[str, ...],
    use_llm: bool,
    llm_provider: str | None,
    llm_base_url: str | None,
    llm_model: str | None,
) -> None:
    """Run the static auditor on ``script_path``."""
    project = _try_load_project()

    with console.status("[cyan]Auditing script...[/cyan]", spinner="dots"):
        result = audit_script(script_path, skip_rules=set(skip_rules))

    render_audit(console, result)

    priorities: dict[str, Any] | None = None
    if use_llm:
        priorities = _maybe_prioritize(
            result,
            model=llm_model,
            provider=llm_provider,
            base_url=llm_base_url,
        )
        if priorities is not None:
            render_audit_priorities(console, priorities)

    if project is not None:
        _persist_audit_result(
            project=project,
            script_path=script_path,
            result=result,
            priorities=priorities,
        )


def _maybe_prioritize(
    result: dict[str, Any],
    *,
    model: str | None,
    provider: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any] | None:
    if not result.get("findings"):
        console.print("\n[dim](--llm: nothing to prioritize, no findings)[/dim]")
        return None
    if not _llm_available("prioritizer", provider, base_url, model):
        return None
    try:
        with console.status("[cyan]Asking the prioritizer...[/cyan]", spinner="dots"):
            return _audit_prioritizer_callable(
                result,
                model=model,
                provider=provider,
                base_url=base_url,
            )
    except AuditAgentError as exc:
        console.print(f"\n[red]✗ Prioritizer returned bad response:[/red] {exc}")
        return None


def _persist_audit_result(
    *,
    project: ProjectContext,
    script_path: Path,
    result: dict[str, Any],
    priorities: dict[str, Any] | None = None,
) -> None:
    """Record an audit run in the project context and advice log."""
    counts: dict[str, int] = {}
    for f in result.get("findings", []):
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    summary_parts = [f"{n} {sev}" for sev, n in counts.items()] or ["no issues"]
    summary = "Audit: " + ", ".join(summary_parts)

    project.append_decision(
        command="audit",
        summary=summary,
        reasoning=(
            f"Script: {script_path}, frameworks: "
            f"{', '.join(result.get('frameworks') or []) or 'none'}"
        ),
    )

    log_path = project.path / "advice.log"
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": "audit",
        "script": str(script_path),
        "frameworks": result.get("frameworks"),
        "findings": result.get("findings"),
    }
    if priorities is not None:
        entry["priorities"] = priorities
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    console.print(f"\n[green]✓[/green] Saved to {project.path}")


# --------------------------------------------------------------------------- #
# watch command                                                               #
# --------------------------------------------------------------------------- #


@cli.command(
    help=(
        "Monitor a training log for plateau / overfit / NaN / divergence. "
        "Accepts plain-text logs or TensorBoard event files/directories."
    )
)
@click.argument(
    "log_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=Path),
)
@click.option(
    "-f",
    "--follow",
    is_flag=True,
    help="Tail the file and report new findings as they appear (Ctrl+C to stop).",
)
@click.option(
    "--poll-interval",
    type=float,
    default=1.0,
    show_default=True,
    help="Seconds between polls when --follow is active.",
)
@click.option(
    "--llm",
    "use_llm",
    is_flag=True,
    help="After the deterministic anomalies, ask Claude to diagnose them.",
)
@llm_target_options
@click.option(
    "--model",
    "llm_model",
    default=None,
    help=f"Model used by the diagnostician when --llm is set. {MODEL_HELP_SUFFIX}",
)
@click.option(
    "--apply",
    "apply_flag",
    is_flag=True,
    help=(
        "After --llm, walk the diagnostician's suggested_edits and apply "
        "each one the user confirms to the config file given by --config."
    ),
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML or JSON config file --apply will edit (required with --apply).",
)
@click.option(
    "-y",
    "--yes",
    "auto_yes",
    is_flag=True,
    help="Apply every proposed edit without prompting (use with care).",
)
def watch(
    log_path: Path,
    follow: bool,
    poll_interval: float,
    use_llm: bool,
    llm_provider: str | None,
    llm_base_url: str | None,
    llm_model: str | None,
    apply_flag: bool,
    config_path: Path | None,
    auto_yes: bool,
) -> None:
    """Run the watch rules over ``log_path``."""
    project = _try_load_project()

    source, raw_snapshots = load_snapshots(log_path)
    snapshots = merge_consecutive_same_epoch(raw_snapshots)
    findings = [f.to_dict() for f in run_all_detectors(snapshots)]
    if source == "tensorboard":
        console.print(f"[dim]Source: TensorBoard event file(s) at {log_path}[/dim]\n")
    elif source == "wandb":
        console.print(f"[dim]Source: W&B run cache at {log_path}[/dim]\n")

    render_watch_report(
        console,
        log_path=str(log_path),
        snapshots=snapshots,
        findings=findings,
    )

    diagnosis: dict[str, Any] | None = None
    apply_result: ApplyResult | None = None
    if use_llm:
        diagnosis = _maybe_diagnose(
            snapshots,
            findings,
            model=llm_model,
            provider=llm_provider,
            base_url=llm_base_url,
        )
        if diagnosis is not None:
            render_watch_diagnosis(console, diagnosis)

    if apply_flag:
        apply_result = _maybe_apply_edits(
            diagnosis=diagnosis,
            config_path=config_path,
            auto_yes=auto_yes,
        )

    if project is not None:
        _persist_watch_result(
            project=project,
            log_path=log_path,
            snapshots=snapshots,
            findings=findings,
            diagnosis=diagnosis,
            apply_result=apply_result,
        )

    if follow:
        if source != "plain_text":
            console.print(
                f"\n[yellow]⚠ --follow is only supported for plain-text "
                f"logs in v0.2 (got source: {source}).[/yellow]"
            )
            return
        try:
            _follow_loop(
                log_path=log_path,
                seen_signatures={_finding_signature(f) for f in findings},
                snapshots=snapshots,
                poll_interval=poll_interval,
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Stopped watching.[/dim]")


def _maybe_diagnose(
    snapshots: list[Any],
    findings: list[dict[str, Any]],
    *,
    model: str | None,
    provider: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any] | None:
    if not findings:
        console.print("\n[dim](--llm: nothing to diagnose, no anomalies)[/dim]")
        return None
    if not _llm_available("diagnostician", provider, base_url, model):
        return None
    snap_payload = [
        {
            "epoch": s.epoch,
            "step": s.step,
            "metrics": s.metrics,
        }
        for s in snapshots
    ]
    try:
        with console.status("[cyan]Asking the diagnostician...[/cyan]", spinner="dots"):
            return _watch_diagnostician_callable(
                snap_payload,
                findings,
                model=model,
                provider=provider,
                base_url=base_url,
            )
    except WatchAgentError as exc:
        console.print(f"\n[red]✗ Diagnostician returned bad response:[/red] {exc}")
        return None


def _maybe_apply_edits(
    *,
    diagnosis: dict[str, Any] | None,
    config_path: Path | None,
    auto_yes: bool,
) -> ApplyResult | None:
    """Drive the permission-gated edit session when ``--apply`` is set."""
    if diagnosis is None:
        console.print(
            "\n[yellow]⚠ --apply needs --llm (and a diagnosis) to know what to "
            "edit; skipping.[/yellow]"
        )
        return None
    if config_path is None:
        console.print("\n[yellow]⚠ --apply requires --config <file>; skipping.[/yellow]")
        return None

    raw_edits = diagnosis.get("suggested_edits") or []
    if not raw_edits:
        console.print(
            "\n[dim]--apply: diagnostician didn't produce any suggested_edits, nothing to do.[/dim]"
        )
        return None

    confirm_fn = make_console_confirm(console, auto_yes=auto_yes)
    try:
        result = _apply_edits_callable(
            config_path,
            raw_edits,
            confirm_fn=confirm_fn,
        )
    except ConfigEditError as exc:
        console.print(f"\n[red]✗ Config edit failed:[/red] {exc}")
        return None

    render_apply_summary(console, result)
    return result


def _follow_loop(
    *,
    log_path: Path,
    seen_signatures: set[str],
    snapshots: list[Any],
    poll_interval: float,
) -> None:
    """Continuously poll ``log_path`` and surface new findings."""
    import time

    last_size = log_path.stat().st_size
    console.print(f"\n[dim]Watching {log_path}... (Ctrl+C to stop)[/dim]")

    while True:
        time.sleep(poll_interval)

        size = log_path.stat().st_size
        if size <= last_size:
            continue

        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(last_size)
            new_text = f.read()
        last_size = size

        new_snapshots = merge_consecutive_same_epoch(snapshots + _parse_text_snapshots(new_text))
        snapshots[:] = new_snapshots

        new_findings = [
            f.to_dict()
            for f in run_all_detectors(snapshots)
            if _finding_signature(f.to_dict()) not in seen_signatures
        ]
        for finding in new_findings:
            seen_signatures.add(_finding_signature(finding))

        if new_findings:
            render_new_findings(console, new_findings)


def _parse_text_snapshots(text: str) -> list[Any]:
    from .tools.logs import parse_log_text

    return parse_log_text(text)


def _finding_signature(finding: dict[str, Any]) -> str:
    """Stable identifier so we don't re-report the same finding."""
    return f"{finding['rule_id']}::{finding.get('epoch')}::{finding['message']}"


def _persist_watch_result(
    *,
    project: ProjectContext,
    log_path: Path,
    snapshots: list[Any],
    findings: list[dict[str, Any]],
    diagnosis: dict[str, Any] | None = None,
    apply_result: ApplyResult | None = None,
) -> None:
    """Record a watch invocation in the project context + advice log."""
    counts: dict[str, int] = {}
    for f in findings:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    summary_parts = [f"{n} {sev}" for sev, n in counts.items()] or ["no issues"]
    summary = f"Watch ({log_path.name}): " + ", ".join(summary_parts)
    if apply_result is not None and apply_result.applied:
        summary += f"; applied {len(apply_result.applied)} edit(s)"

    project.append_decision(
        command="watch",
        summary=summary,
        reasoning=f"Parsed {len(snapshots)} snapshots from {log_path}",
    )

    log_entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": "watch",
        "log": str(log_path),
        "snapshots": len(snapshots),
        "findings": findings,
    }
    if diagnosis is not None:
        log_entry["diagnosis"] = diagnosis
    if apply_result is not None:
        log_entry["applied_edits"] = [
            {
                "key": e.key,
                "from": e.current_value,
                "to": e.proposed_value,
                "rationale": e.rationale,
            }
            for e in apply_result.applied
        ]
        log_entry["rejected_edits"] = [
            {"key": e.key, "from": e.current_value, "to": e.proposed_value}
            for e in apply_result.rejected
        ]
        if apply_result.backup_path is not None:
            log_entry["backup"] = str(apply_result.backup_path)
    with (project.path / "advice.log").open("a", encoding="utf-8") as out:
        out.write(json.dumps(log_entry) + "\n")


# --------------------------------------------------------------------------- #
# compare command                                                             #
# --------------------------------------------------------------------------- #


@cli.command(
    help="Compare two training runs side-by-side (config + final metrics).",
)
@click.argument("run_a")
@click.argument("run_b")
@click.option(
    "--llm",
    "use_llm",
    is_flag=True,
    help="After the deterministic diff, ask Claude to explain why the winner won.",
)
@llm_target_options
@click.option(
    "--model",
    "llm_model",
    default=None,
    help=f"Model used by the hypothesizer when --llm is set. {MODEL_HELP_SUFFIX}",
)
def compare(
    run_a: str,
    run_b: str,
    use_llm: bool,
    llm_provider: str | None,
    llm_base_url: str | None,
    llm_model: str | None,
) -> None:
    """Compare two runs by identifier or directory path."""
    project = _try_load_project()

    try:
        record_a = load_run(run_a, project=project)
        record_b = load_run(run_b, project=project)
    except RunNotFoundError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise SystemExit(1) from exc

    comparison = compare_runs(record_a, record_b)
    render_compare(console, comparison)

    hypothesis: dict[str, Any] | None = None
    if use_llm:
        hypothesis = _maybe_hypothesize(
            comparison,
            model=llm_model,
            provider=llm_provider,
            base_url=llm_base_url,
        )
        if hypothesis is not None:
            render_compare_hypothesis(console, hypothesis)

    if project is not None:
        _persist_compare_result(
            project=project,
            run_a=record_a.id,
            run_b=record_b.id,
            comparison=comparison,
            hypothesis=hypothesis,
        )


def _maybe_hypothesize(
    comparison: dict[str, Any],
    *,
    model: str | None,
    provider: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any] | None:
    if not _llm_available("hypothesizer", provider, base_url, model):
        return None
    try:
        with console.status("[cyan]Asking the hypothesizer...[/cyan]", spinner="dots"):
            return _compare_hypothesizer_callable(
                comparison,
                model=model,
                provider=provider,
                base_url=base_url,
            )
    except CompareAgentError as exc:
        console.print(f"\n[red]✗ Hypothesizer returned bad response:[/red] {exc}")
        return None


def _persist_compare_result(
    *,
    project: ProjectContext,
    run_a: str,
    run_b: str,
    comparison: dict[str, Any],
    hypothesis: dict[str, Any] | None = None,
) -> None:
    """Record a compare invocation in the project's decision log."""
    verdict = comparison.get("verdict", "inconclusive")
    summary = f"Compare {run_a} vs {run_b}: {verdict}"
    project.append_decision(
        command="compare",
        summary=summary,
        reasoning=comparison.get("verdict_explanation", ""),
    )

    log_path = project.path / "advice.log"
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": "compare",
        "run_a": run_a,
        "run_b": run_b,
        "verdict": verdict,
        "verdict_explanation": comparison.get("verdict_explanation"),
        "config_diff": comparison.get("config_diff"),
        "metric_comparison": comparison.get("metric_comparison"),
    }
    if hypothesis is not None:
        entry["hypothesis"] = hypothesis
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------- #
# evaluate command                                                            #
# --------------------------------------------------------------------------- #


@cli.command(
    help="Run post-training analysis on a predictions table.",
)
@click.argument(
    "results_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--y-true", "y_true_col", default=None, help="Ground-truth column.")
@click.option("--y-pred", "y_pred_col", default=None, help="Predicted-label column.")
@click.option("--y-prob", "y_prob_col", default=None, help="Predicted-probability column (binary).")
@click.option(
    "--task",
    type=click.Choice(
        ["binary_classification", "multiclass_classification", "regression"],
    ),
    default=None,
    help="Force a task type (otherwise inferred from the data).",
)
@click.option(
    "--hard-examples",
    "hard_examples_k",
    type=int,
    default=5,
    show_default=True,
    help="How many worst-error rows to surface.",
)
@click.option(
    "--llm",
    "use_llm",
    is_flag=True,
    help="After the deterministic report, ask Claude to interpret the results.",
)
@llm_target_options
@click.option(
    "--model",
    "llm_model",
    default=None,
    help=f"Model used by the interpreter when --llm is set. {MODEL_HELP_SUFFIX}",
)
def evaluate(
    results_path: Path,
    y_true_col: str | None,
    y_pred_col: str | None,
    y_prob_col: str | None,
    task: str | None,
    hard_examples_k: int,
    use_llm: bool,
    llm_provider: str | None,
    llm_base_url: str | None,
    llm_model: str | None,
) -> None:
    """Evaluate predictions in ``results_path``."""
    project = _try_load_project()

    try:
        with console.status("[cyan]Loading predictions...[/cyan]", spinner="dots"):
            df = load_results(results_path)
        with console.status("[cyan]Running evaluation...[/cyan]", spinner="dots"):
            result = run_evaluation(
                df,
                y_true_col=y_true_col,
                y_pred_col=y_pred_col,
                y_prob_col=y_prob_col,
                task=task,
                hard_examples_k=hard_examples_k,
            )
    except EvaluationError as exc:
        console.print(f"[red]✗ Evaluation failed:[/red] {exc}")
        raise SystemExit(2) from exc

    render_evaluation(console, result)

    interpretation: dict[str, Any] | None = None
    leakage_narration: dict[str, Any] | None = None
    if use_llm:
        interpretation = _maybe_interpret_evaluation(
            result,
            model=llm_model,
            provider=llm_provider,
            base_url=llm_base_url,
        )
        if interpretation is not None:
            render_evaluation_interpretation(console, interpretation)
        # v0.7: when evaluate auto-attached leakage evidence (smell
        # threshold fired), narrate it with the anti-hallucination
        # investigator agent.
        if result.get("leakage_investigation"):
            leakage_narration = _maybe_investigate_leakage(
                result["leakage_investigation"],
                model=llm_model,
                provider=llm_provider,
                base_url=llm_base_url,
            )
            if leakage_narration is not None:
                render_leakage_narration(console, leakage_narration)

    if project is not None:
        _persist_evaluate_result(
            project=project,
            results_path=results_path,
            result=result,
            interpretation=interpretation,
        )


def _maybe_investigate_leakage(
    evidence: dict[str, Any],
    *,
    model: str | None,
    provider: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any] | None:
    """Run the leakage investigator if a provider is reachable."""
    if not _has_api_key(provider, base_url):
        console.print(
            f"\n[yellow]⚠ Leakage smell fired but "
            f"{_llm_unavailable_reason(provider, base_url)}; "
            "skipping investigator narration. The evidence panel above is "
            "still the deterministic ground truth.[/yellow]"
        )
        return None
    try:
        with console.status("[cyan]Asking the leakage investigator...[/cyan]", spinner="dots"):
            return _leakage_investigator_callable(
                evidence,
                model=model,
                provider=provider,
                base_url=base_url,
            )
    except LeakageAgentError as exc:
        console.print(f"\n[red]✗ Leakage investigator returned malformed response:[/red] {exc}")
        return None


def _default_leakage_investigator(
    evidence: dict[str, Any],
    *,
    model: str | None,
    provider: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Call the published evidence-bound investigator, unmodified.

    ``agents/leakage_investigator.py`` is the artifact the paper measures,
    so it is not touched here. It already takes ``provider`` and an injected
    ``client``; it just has no ``base_url`` parameter of its own. So when
    the user picks a non-default endpoint we build the client on this side
    and hand it over. With no flags set, ``client`` stays None and
    ``provider`` stays "anthropic" — exactly the call it received before.
    """
    kwargs: dict[str, Any] = {}
    if model is not None:
        kwargs["model"] = model

    resolved_provider, resolved_base_url = _llm_target(provider, base_url)
    if resolved_provider != PROVIDER_ANTHROPIC or resolved_base_url:
        config = resolve_llm_config(
            default_model=LEAKAGE_MODEL_DEFAULT,
            provider=provider,
            model=model,
            base_url=base_url,
        )
        kwargs["provider"] = config.provider
        kwargs["model"] = config.model
        kwargs["client"] = build_client(config)

    # Production path: the evidence-bound runtime-schema contract. The
    # narrator answers through a tool whose ``columns_referenced`` enum is
    # generated from this evidence dict at call time, and the cited columns
    # are deterministically re-validated against the evidence set. The prose
    # ``investigate_leakage`` remains available as a tool-free fallback.
    return investigate_leakage_bound(evidence, **kwargs)


_leakage_investigator_callable: Callable[..., dict[str, Any]] = _default_leakage_investigator


def _maybe_interpret_evaluation(
    result: dict[str, Any],
    *,
    model: str | None,
    provider: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any] | None:
    if not _llm_available("interpreter", provider, base_url, model):
        return None
    try:
        with console.status("[cyan]Asking the interpreter...[/cyan]", spinner="dots"):
            return _evaluate_interpreter_callable(
                result,
                model=model,
                provider=provider,
                base_url=base_url,
            )
    except EvaluateAgentError as exc:
        console.print(f"\n[red]✗ Interpreter returned bad response:[/red] {exc}")
        return None


def _persist_evaluate_result(
    *,
    project: ProjectContext,
    results_path: Path,
    result: dict[str, Any],
    interpretation: dict[str, Any] | None = None,
) -> None:
    """Record an evaluate invocation in the project context + advice log."""
    task = result.get("task", "unknown")
    metrics = result.get("metrics") or {}
    headline = (
        ", ".join(f"{name}={value}" for name, value in list(metrics.items())[:3]) or "no metrics"
    )
    summary = f"Evaluate ({task}): {headline}"

    project.append_decision(
        command="evaluate",
        summary=summary,
        reasoning=f"Results: {results_path}; rows: {result.get('rows', 0)}",
    )

    log_entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": "evaluate",
        "results": str(results_path),
        "task": task,
        "metrics": metrics,
        "warnings": result.get("warnings", []),
    }
    if interpretation is not None:
        log_entry["interpretation"] = interpretation
    with (project.path / "advice.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


# --------------------------------------------------------------------------- #
# deploy command                                                              #
# --------------------------------------------------------------------------- #


@cli.command(
    help="Run a deployment-readiness check on a saved model file.",
)
@click.argument(
    "model_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--requirements",
    "requirements_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a requirements.txt / pyproject.toml / environment.yml.",
)
@click.option(
    "--target",
    type=click.Choice(list(DEPLOY_TARGETS)),
    default="local",
    show_default=True,
    help="Deployment target name.",
)
@click.option(
    "--llm",
    "use_llm",
    is_flag=True,
    help="After the deterministic checks, ask Claude for a production verdict.",
)
@llm_target_options
@click.option(
    "--model",
    "llm_model",
    default=None,
    help=f"Model used by the advisor when --llm is set. {MODEL_HELP_SUFFIX}",
)
def deploy(
    model_path: Path,
    requirements_path: Path | None,
    target: str,
    use_llm: bool,
    llm_provider: str | None,
    llm_base_url: str | None,
    llm_model: str | None,
) -> None:
    """Inspect a model file and produce a deployment-readiness report."""
    project = _try_load_project()

    try:
        with console.status("[cyan]Inspecting model...[/cyan]", spinner="dots"):
            report = assess_deployment(
                model_path,
                requirements_path=requirements_path,
                target=target,
            )
    except DeployAnalysisError as exc:
        console.print(f"[red]✗ Deploy check failed:[/red] {exc}")
        raise SystemExit(2) from exc

    render_deployment(console, report)

    advice: dict[str, Any] | None = None
    if use_llm:
        advice = _maybe_advise_deployment(
            report,
            model=llm_model,
            provider=llm_provider,
            base_url=llm_base_url,
        )
        if advice is not None:
            render_deployment_advice(console, advice)

    if project is not None:
        _persist_deploy_result(
            project=project,
            model_path=model_path,
            requirements_path=requirements_path,
            target=target,
            report=report,
            advice=advice,
        )


def _maybe_advise_deployment(
    report: dict[str, Any],
    *,
    model: str | None,
    provider: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any] | None:
    if not _llm_available("advisor", provider, base_url, model):
        return None
    try:
        with console.status("[cyan]Asking the advisor...[/cyan]", spinner="dots"):
            return _deploy_advisor_callable(
                report,
                model=model,
                provider=provider,
                base_url=base_url,
            )
    except DeployAgentError as exc:
        console.print(f"\n[red]✗ Advisor returned bad response:[/red] {exc}")
        return None


def _persist_deploy_result(
    *,
    project: ProjectContext,
    model_path: Path,
    requirements_path: Path | None,
    target: str,
    report: dict[str, Any],
    advice: dict[str, Any] | None = None,
) -> None:
    """Record a deploy invocation in the project context + advice log."""
    model = report.get("model") or {}
    n_warn = len(report.get("warnings") or [])
    summary = (
        f"Deploy ({target}, {model.get('format', '?')}, "
        f"{model.get('size_pretty', '?')}): " + (f"{n_warn} warning(s)" if n_warn else "clean")
    )

    project.append_decision(
        command="deploy",
        summary=summary,
        reasoning=f"Model: {model_path}; target: {target}",
    )

    log_entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": "deploy",
        "model": str(model_path),
        "requirements": str(requirements_path) if requirements_path else None,
        "target": target,
        "format": model.get("format"),
        "size_bytes": model.get("size_bytes"),
        "warnings": report.get("warnings", []),
        "checklist": report.get("checklist", []),
    }
    if advice is not None:
        log_entry["advice"] = advice
    with (project.path / "advice.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


# --------------------------------------------------------------------------- #
# status command                                                              #
# --------------------------------------------------------------------------- #


@cli.command(
    help="Print a summary of the active mlcompass project context.",
)
@click.option(
    "--recent",
    "recent_decisions",
    type=int,
    default=5,
    show_default=True,
    help="How many of the latest decisions to surface.",
)
def status(recent_decisions: int) -> None:
    """Show a structured snapshot of the active project."""
    try:
        project = ProjectContext.load()
    except ProjectNotFoundError as exc:
        console.print(
            "[red]✗[/red] No .mlcompass/ project found at or above the "
            "current directory. Run `mlcompass init <name>` to create one."
        )
        raise SystemExit(1) from exc

    render_status(console, project, recent_decisions=recent_decisions)


# --------------------------------------------------------------------------- #
# monitor — post-deploy drift detection (Faz 8a)                              #
# --------------------------------------------------------------------------- #


@cli.command(
    help="Compare a reference dataset to a current one and surface drift.",
)
@click.argument(
    "reference_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "current_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--features",
    default=None,
    help=("Comma-separated list of columns to inspect. Default: all overlapping columns."),
)
@click.option(
    "--bins",
    type=int,
    default=10,
    show_default=True,
    help="Number of bins for PSI / chi-square on numeric columns.",
)
@click.option(
    "--top",
    "top_n",
    type=int,
    default=5,
    show_default=True,
    help="How many most-drifted features to highlight.",
)
@click.option(
    "--llm",
    "use_llm",
    is_flag=True,
    help="After the deterministic report, ask Claude to interpret the drift.",
)
@llm_target_options
@click.option(
    "--model",
    "llm_model",
    default=None,
    help=f"Model used by the interpreter when --llm is set. {MODEL_HELP_SUFFIX}",
)
def monitor(
    reference_path: Path,
    current_path: Path,
    features: str | None,
    bins: int,
    top_n: int,
    use_llm: bool,
    llm_provider: str | None,
    llm_base_url: str | None,
    llm_model: str | None,
) -> None:
    """Run drift detection between two datasets."""
    project = _try_load_project()

    feature_list = [f.strip() for f in features.split(",") if f.strip()] if features else None

    try:
        with console.status("[cyan]Loading reference + current...[/cyan]", spinner="dots"):
            ref_df = load_drift_table(reference_path)
            cur_df = load_drift_table(current_path)
        with console.status("[cyan]Running drift checks...[/cyan]", spinner="dots"):
            result = detect_drift(
                ref_df,
                cur_df,
                features=feature_list,
                bins=bins,
                top_n=top_n,
            )
    except DriftAnalysisError as exc:
        console.print(f"[red]✗ Drift check failed:[/red] {exc}")
        raise SystemExit(2) from exc

    render_drift(console, result)

    interpretation: dict[str, Any] | None = None
    if use_llm:
        interpretation = _maybe_interpret_drift(
            result,
            model=llm_model,
            provider=llm_provider,
            base_url=llm_base_url,
        )
        if interpretation is not None:
            render_drift_interpretation(console, interpretation)

    if project is not None:
        _persist_monitor_result(
            project=project,
            reference_path=reference_path,
            current_path=current_path,
            result=result,
            interpretation=interpretation,
        )

    # Exit non-zero when drift is major so CI / cron pipelines can gate on it.
    if result["verdict"]["status"] == "major_drift":
        raise SystemExit(1)


def _maybe_interpret_drift(
    result: dict[str, Any],
    *,
    model: str | None,
    provider: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any] | None:
    if not _llm_available("interpreter", provider, base_url, model):
        return None
    try:
        with console.status("[cyan]Asking the drift interpreter...[/cyan]", spinner="dots"):
            return _monitor_interpreter_callable(
                result,
                model=model,
                provider=provider,
                base_url=base_url,
            )
    except MonitorAgentError as exc:
        console.print(f"\n[red]✗ Interpreter returned bad response:[/red] {exc}")
        return None


# Indirection so tests can monkeypatch without standing up the LLM.
def _default_monitor_interpreter(
    result: dict[str, Any],
    *,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    return interpret_drift(result, model=model, provider=provider, base_url=base_url)


_monitor_interpreter_callable: Callable[..., dict[str, Any]] = _default_monitor_interpreter


def _persist_monitor_result(
    *,
    project: ProjectContext,
    reference_path: Path,
    current_path: Path,
    result: dict[str, Any],
    interpretation: dict[str, Any] | None = None,
) -> None:
    """Record a monitor invocation in the project context + advice log."""
    verdict = result["verdict"]
    agg = result["aggregate"]
    mean_psi = agg.get("mean_psi")
    max_psi = agg.get("max_psi")
    summary = (
        f"Monitor ({verdict['status']}): mean PSI {mean_psi:.3f}, max {max_psi:.3f}"
        if mean_psi is not None and max_psi is not None
        else f"Monitor ({verdict['status']})"
    )
    project.append_decision(
        command="monitor",
        summary=summary,
        reasoning=f"Reference {reference_path.name} vs current {current_path.name}",
    )

    log_entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": "monitor",
        "reference": str(reference_path),
        "current": str(current_path),
        "verdict": verdict,
        "aggregate": agg,
        "top_drifted": result.get("top_drifted", []),
    }
    if interpretation is not None:
        log_entry["interpretation"] = interpretation
    with (project.path / "advice.log").open("a", encoding="utf-8") as out:
        out.write(json.dumps(log_entry) + "\n")


# --------------------------------------------------------------------------- #
# optimize — HPO sub-agent over the run history (Faz 8b)                      #
# --------------------------------------------------------------------------- #


@cli.command(
    help="Analyse a run history and suggest next hyperparameter configs.",
)
@click.option(
    "--runs-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "Directory holding run subfolders. Defaults to "
        "<project>/runs/ if an mlcompass project is active."
    ),
)
@click.option(
    "--metric",
    required=True,
    help="Metric to optimize (case-insensitive, e.g. val_acc).",
)
@click.option(
    "--direction",
    type=click.Choice(["max", "min"]),
    default="max",
    show_default=True,
    help="Whether to maximise or minimise the metric.",
)
@click.option(
    "--top",
    "top_k",
    type=int,
    default=5,
    show_default=True,
    help="Leaderboard size.",
)
@click.option(
    "--suggestions",
    "n_suggestions",
    type=int,
    default=3,
    show_default=True,
    help="How many next-config recommendations to emit.",
)
@click.option(
    "--constraints",
    default=None,
    help=("Bounds on numeric hyperparameters, comma-separated: 'lr:0.0001-0.1,batch_size:16-256'."),
)
@click.option(
    "--llm",
    "use_llm",
    is_flag=True,
    help="After the deterministic report, ask Claude for a strategist plan.",
)
@llm_target_options
@click.option(
    "--model",
    "llm_model",
    default=None,
    help=f"Model used by the strategist when --llm is set. {MODEL_HELP_SUFFIX}",
)
def optimize(
    runs_dir: Path | None,
    metric: str,
    direction: str,
    top_k: int,
    n_suggestions: int,
    constraints: str | None,
    use_llm: bool,
    llm_provider: str | None,
    llm_base_url: str | None,
    llm_model: str | None,
) -> None:
    """Recommend the next hyperparameter configurations."""
    project = _try_load_project()
    if runs_dir is None:
        if project is None:
            console.print(
                "[red]✗[/red] No --runs-dir supplied and no mlcompass "
                "project found. Run `mlcompass init <name>` or pass "
                "--runs-dir explicitly."
            )
            raise SystemExit(2)
        runs_dir = project.path / "runs"
    if not runs_dir.is_dir():
        console.print(f"[red]✗[/red] Runs directory {runs_dir} does not exist.")
        raise SystemExit(2)

    try:
        constraint_map = parse_constraints(constraints or "")
    except OptimizeAnalysisError as exc:
        console.print(f"[red]✗ Bad --constraints:[/red] {exc}")
        raise SystemExit(2) from exc

    with console.status(f"[cyan]Loading runs from {runs_dir}...[/cyan]", spinner="dots"):
        runs = load_runs_from_dir(runs_dir)

    if not runs:
        console.print(f"[red]✗[/red] No valid run directories found under {runs_dir}.")
        raise SystemExit(2)

    try:
        with console.status("[cyan]Optimizing...[/cyan]", spinner="dots"):
            result = optimize_history(
                runs,
                metric=metric,
                direction=direction,
                top_k=top_k,
                n_suggestions=n_suggestions,
                constraints=constraint_map,
            )
    except OptimizeAnalysisError as exc:
        console.print(f"[red]✗ Optimize failed:[/red] {exc}")
        raise SystemExit(2) from exc

    render_optimize(console, result)

    strategy: dict[str, Any] | None = None
    if use_llm:
        strategy = _maybe_strategize_optimize(
            result,
            model=llm_model,
            provider=llm_provider,
            base_url=llm_base_url,
        )
        if strategy is not None:
            render_optimize_strategy(console, strategy)

    if project is not None:
        _persist_optimize_result(
            project=project,
            runs_dir=runs_dir,
            result=result,
            strategy=strategy,
        )


def _maybe_strategize_optimize(
    result: dict[str, Any],
    *,
    model: str | None,
    provider: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any] | None:
    if not _llm_available("strategist", provider, base_url, model):
        return None
    try:
        with console.status("[cyan]Asking the HPO strategist...[/cyan]", spinner="dots"):
            return _optimize_strategist_callable(
                result,
                model=model,
                provider=provider,
                base_url=base_url,
            )
    except OptimizeAgentError as exc:
        console.print(f"\n[red]✗ Strategist returned bad response:[/red] {exc}")
        return None


# Indirection for tests.
def _default_optimize_strategist(
    result: dict[str, Any],
    *,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    return strategize_optimize(result, model=model, provider=provider, base_url=base_url)


_optimize_strategist_callable: Callable[..., dict[str, Any]] = _default_optimize_strategist


def _persist_optimize_result(
    *,
    project: ProjectContext,
    runs_dir: Path,
    result: dict[str, Any],
    strategy: dict[str, Any] | None = None,
) -> None:
    """Record an optimize invocation in the project context + advice log."""
    best = result["best"]
    summary = (
        f"Optimize ({result['metric']} {result['direction']}): best "
        f"{best['id']} @ {best['score']:.4g}"
    )
    project.append_decision(
        command="optimize",
        summary=summary,
        reasoning=(
            f"Runs analysed: {result['n_scored']}/{result['n_runs']}; "
            f"{len(result.get('suggestions', []))} suggestions."
        ),
    )
    log_entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": "optimize",
        "runs_dir": str(runs_dir),
        "metric": result["metric"],
        "direction": result["direction"],
        "best": best,
        "sensitivity": result["sensitivity"],
        "suggestions": result["suggestions"],
    }
    if strategy is not None:
        log_entry["strategy"] = strategy
    with (project.path / "advice.log").open("a", encoding="utf-8") as out:
        out.write(json.dumps(log_entry) + "\n")


# --------------------------------------------------------------------------- #
# agent — self-driving over the eight tools (Faz 7)                           #
# --------------------------------------------------------------------------- #


@cli.command(help="Run a self-driving agent over the eight mlcompass tools.")
@click.argument("task")
@click.option(
    "--project-path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("."),
    show_default=True,
    help=(
        "Path inside or above an mlcompass project. The transcript is "
        "written under .mlcompass/agent_runs/<id>/."
    ),
)
@click.option(
    "--backend",
    type=click.Choice(["api", "claude-code"]),
    default="api",
    show_default=True,
    help=(
        "Which driver to use. 'api' calls Anthropic directly (needs "
        "ANTHROPIC_API_KEY). 'claude-code' routes through your local "
        "Claude Code CLI via the claude-agent-sdk."
    ),
)
@click.option(
    "--model",
    default=None,
    help=(
        "Model name. Defaults to claude-sonnet-4-5 (override via MLCOMPASS_AGENT_MODEL env var)."
    ),
)
@click.option(
    "--max-turns",
    type=int,
    default=None,
    help="Hard cap on conversation turns (default: 20).",
)
@click.option(
    "--auto-approve",
    is_flag=True,
    help=(
        "Skip the y/N permission prompt before mutating tools. Use for "
        "CI / headless runs where no human is present."
    ),
)
@click.option(
    "--resume",
    "resume_from",
    default=None,
    help=(
        "Resume from a prior agent run ID (subdirectory under "
        ".mlcompass/agent_runs/). The orchestrator summarises the "
        "prior transcript and prepends it to the new task."
    ),
)
@click.option(
    "--no-memory",
    "no_memory",
    is_flag=True,
    help=(
        "Skip cross-session memory injection. The agent starts cold, "
        "without seeing prior decisions or the last agent run."
    ),
)
def agent(
    task: str,
    project_path: Path,
    backend: str,
    model: str | None,
    max_turns: int | None,
    auto_approve: bool,
    resume_from: str | None,
    no_memory: bool,
) -> None:
    """Drive the chosen backend until it answers or hits max-turns."""
    # Lazy-import the orchestrator's defaults so users without
    # `anthropic` / `claude-agent-sdk` installed can still run the
    # rest of the CLI. The real driver call goes through `_agent_runner`
    # (a tiny indirection so tests can swap it without spinning up the
    # SDK), not through this import.
    try:
        from .agent.orchestrator import (
            DEFAULT_MAX_TURNS,
            DEFAULT_MODEL,
        )
    except ImportError as exc:  # pragma: no cover
        console.print(
            "[red]✗[/red] The agent command needs an extra. "
            "Install with: pip install 'mlcompass[agent]' "
            "(or pip install 'mlcompass[agent-claude-code]' for the "
            "claude-code backend)."
        )
        raise SystemExit(2) from exc

    # The "api" backend talks directly to the Anthropic API and needs
    # ANTHROPIC_API_KEY. Without it the SDK throws a long auth error
    # several turns into the conversation (field-test bug #6). Catch
    # the missing-key case up front so the user sees a clean message
    # and exits with a stable code instead of an opaque api_error.
    # The "claude-code" backend has its own auth flow inside the
    # Claude Code CLI and does NOT require ANTHROPIC_API_KEY.
    if backend == "api" and not _has_api_key():
        console.print(
            "[red]✗[/red] The 'api' backend requires ANTHROPIC_API_KEY "
            "to be set in your environment.\n"
            "Either:\n"
            "  • [bold]set the key[/bold]: "
            "`export ANTHROPIC_API_KEY=sk-ant-...` "
            '(or `$env:ANTHROPIC_API_KEY = "..."` in PowerShell), or\n'
            "  • [bold]switch backends[/bold]: pass `--backend claude-code` "
            "to route through your local Claude Code CLI."
        )
        raise SystemExit(2)

    summary = _agent_runner(
        task=task,
        project_path=str(project_path),
        backend=backend,
        model=model or DEFAULT_MODEL,
        max_turns=max_turns or DEFAULT_MAX_TURNS,
        auto_approve_mutations=auto_approve,
        resume_from=resume_from,
        use_memory=not no_memory,
    )

    console.print()
    console.print(
        Panel.fit(
            (
                f"Transcript: [cyan]{summary.transcript_path}[/cyan]\n"
                f"Summary:    [cyan]{summary.summary_path}[/cyan]\n"
                f"Backend:    [bold]{backend}[/bold]   "
                f"Turns: [bold]{summary.result.turns}[/bold]   "
                f"Stop: [bold]{summary.result.stop_reason}[/bold]"
            ),
            title="📋 Agent run",
            border_style="cyan",
        )
    )

    if not summary.result.ok:
        raise SystemExit(1)


# Indirection so tests can monkeypatch the orchestrator without
# bringing the real Anthropic / claude-agent-sdk SDKs online.
def _default_agent_runner(**kwargs: Any) -> Any:
    from .agent.orchestrator import run_agent

    return run_agent(**kwargs)


_agent_runner: Callable[..., Any] = _default_agent_runner


# --------------------------------------------------------------------------- #
# install-slash-commands — Claude Code slash-command bootstrapper (Faz 10)    #
# --------------------------------------------------------------------------- #


@cli.command(
    "install-slash-commands",
    help="Install mlcompass Claude Code slash commands to .claude/commands/.",
)
@click.option(
    "--scope",
    type=click.Choice(["project", "user"]),
    default="project",
    show_default=True,
    help=(
        "Where to install: 'project' = .claude/commands/ in cwd "
        "(loads only when you cd in); 'user' = ~/.claude/commands/ "
        "(loads everywhere)."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing files with the same name.",
)
def install_slash_commands(scope: str, force: bool) -> None:
    """Copy mlcompass's shipped slash-command .md files into Claude Code's
    commands directory so the user gets ``/mlc-advise``, ``/mlc-leak``,
    ``/mlc-evaluate``, … as first-class slash commands.

    All eleven commands are prefixed ``mlc-`` to avoid colliding with
    MCP tool names (Claude Code's slash menu would otherwise route
    ``/advise`` to the MCP tool browser instead of firing the command).
    """
    target = (
        Path.home() / ".claude" / "commands"
        if scope == "user"
        else Path.cwd() / ".claude" / "commands"
    )
    target.mkdir(parents=True, exist_ok=True)

    source = Path(__file__).parent / "slash_commands"
    if not source.is_dir():
        console.print(
            "[red]✗[/red] Could not locate the shipped slash_commands/ "
            "directory inside the mlcompass package. This is likely a "
            "packaging bug — please file an issue."
        )
        raise SystemExit(2)

    md_files = sorted(source.glob("*.md"))
    if not md_files:
        console.print(
            "[red]✗[/red] No slash-command .md files were shipped. This is a packaging bug."
        )
        raise SystemExit(2)

    installed: list[str] = []
    skipped: list[str] = []
    for md in md_files:
        dest = target / md.name
        if dest.exists() and not force:
            skipped.append(md.name)
            continue
        dest.write_text(md.read_text(encoding="utf-8"), encoding="utf-8")
        installed.append(md.name)

    body_lines: list[str] = [
        f"[bold]Target:[/bold] {target}",
        f"[bold]Installed:[/bold] {len(installed)} / {len(md_files)}",
    ]
    if installed:
        commands = ", ".join(f"/{md[:-3]}" for md in installed)
        body_lines.append(f"\n[green]✓ Available now:[/green]\n  {commands}")
    if skipped:
        body_lines.append(
            "\n[yellow]⚠ Skipped (already exist):[/yellow]\n  "
            + ", ".join(f"/{md[:-3]}" for md in skipped)
        )
        body_lines.append("\n[dim]Re-run with --force to overwrite the skipped files.[/dim]")

    body_lines.append(
        "\n[dim]Restart Claude Code (or reload the project) and type "
        "the slash key to see them in the slash menu.[/dim]"
    )

    console.print(
        Panel.fit(
            "\n".join(body_lines),
            title=":sparkles: mlcompass slash commands",
            border_style="cyan",
        )
    )


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main() -> None:
    """Entry point for the ``mlcompass`` console script."""
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
