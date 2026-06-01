"""mlcompass MCP server — expose mlcompass's deterministic tools to MCP clients.

The MCP (Model Context Protocol) server speaks JSON-RPC over stdio, the
transport Claude Desktop and other MCP-capable clients use to discover
and call tools. Install with::

    pip install 'mlcompass[mcp]'

Then point a client at the ``mlcompass-mcp`` script. For Claude Desktop
that's an entry in ``claude_desktop_config.json``::

    {
      "mcpServers": {
        "mlcompass": {
          "command": "mlcompass-mcp"
        }
      }
    }

Design notes:
    - Tools are deterministic. There is no ``--llm`` knob on the MCP
      surface; the calling LLM (Claude) is the interpreter, and asking
      it to call a tool that in turn calls Claude would be both wasteful
      and confusing.
    - All tools are read- or compute-only with one deliberate exception:
      ``mlcompass_init`` creates a ``.mlcompass/`` directory.
    - ``watch --apply`` is intentionally not exposed. Config-mutating
      operations need a confirm channel that MCP doesn't standardise;
      use the CLI for those.
"""

from __future__ import annotations

import contextlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # pragma: no cover — exercised only at import time
    raise ImportError(
        "mlcompass-mcp requires the 'mcp' package. Install with: pip install 'mlcompass[mcp]'"
    ) from e

from . import __version__
from .context import ProjectContext, ProjectExistsError, ProjectNotFoundError
from .tools.anomaly import run_all_detectors
from .tools.dataset import analyze_dataset
from .tools.deploy import (
    DeployAnalysisError,
    assess_deployment,
)
from .tools.evaluation import (
    EvaluationError,
    load_results,
)
from .tools.evaluation import (
    evaluate as run_evaluation,
)
from .tools.logs import load_snapshots, merge_consecutive_same_epoch
from .tools.runs import RunNotFoundError, compare_runs, load_run
from .tools.script import audit_script

mcp = FastMCP(
    "mlcompass",
    instructions=(
        f"mlcompass v{__version__} — your ML pipeline co-pilot. Eight tools "
        "cover dataset analysis, training-script audit, training-log "
        "monitoring, run comparison, post-training evaluation, deployment "
        "readiness, plus project lifecycle (init, status). All tools are "
        "deterministic; you the LLM are the interpreter of their structured "
        "output."
    ),
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _json_safe(value: Any) -> Any:
    """Replace NaN / ±Inf with ``None`` so the return survives JSON encoding.

    Training logs routinely contain ``NaN`` losses (the whole point of
    ``mlcompass_watch`` is to surface them) and the standard JSON spec
    has no representation for them. We map them to ``null`` and rely on
    the structured ``findings`` list to carry the semantic signal.
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def _error(exc: BaseException) -> dict[str, Any]:
    """Uniform error envelope for tool failures."""
    return {"ok": False, "error": type(exc).__name__, "message": str(exc)}


# --------------------------------------------------------------------------- #
# Ledger persistence (v0.7.2)                                                 #
# --------------------------------------------------------------------------- #
#
# Field Test #4 surfaced that the MCP tools were stateless — they
# returned structured output but didn't write to the .mlcompass/
# ledger the way the CLI commands do. That made ``mlcompass_status``
# come back empty even after a full pipeline run through the MCP
# server, which broke the parity contract between the CLI and MCP
# surfaces.
#
# Fix: a small helper that locates the active project (walks up from
# the supplied path) and, if one is found, appends both a decision
# entry and an advice.log line. Missing project ⇒ silent skip — we
# don't want the MCP tool to fail just because the user hasn't run
# ``mlcompass_init`` yet.


def _persist_to_ledger(
    *,
    search_from: str,
    command: str,
    summary: str,
    reasoning: str = "",
    log_extra: dict[str, Any] | None = None,
) -> None:
    """Append a decision + advice.log entry to the active project, if any.

    All exceptions are swallowed deliberately. The ledger is a "nice
    to have" — never the reason a tool fails. The actual analysis the
    user asked for has already happened by the time this is called.
    """
    try:
        project = ProjectContext.load(search_from=search_from)
    except ProjectNotFoundError:
        return
    except Exception:  # noqa: BLE001 — defensive
        return

    # Both writes are best-effort — the ledger is a "nice to have"
    # and never the reason a tool fails. Suppress everything.
    with contextlib.suppress(Exception):
        project.append_decision(
            command=command,
            summary=summary,
            reasoning=reasoning,
        )

    with contextlib.suppress(Exception):
        from datetime import datetime, timezone

        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": command,
        }
        if log_extra:
            entry.update(log_extra)
        with (project.path / "advice.log").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")


def _path_dir(path: str) -> str:
    """Return the parent directory of ``path`` (or ``path`` itself if it's a dir)."""
    p = Path(path)
    if p.is_dir():
        return str(p)
    return str(p.parent or Path("."))


# --------------------------------------------------------------------------- #
# Project lifecycle                                                           #
# --------------------------------------------------------------------------- #


@mcp.tool()
def mlcompass_init(
    name: str,
    parent_dir: str = ".",
    default_model: str = "claude-opus-4-7",
) -> dict[str, Any]:
    """Create a new ``.mlcompass/`` project directory.

    Args:
        name: Human-readable project name (e.g. ``"churn-model"``).
        parent_dir: Directory under which ``.mlcompass/`` is created.
            Defaults to the current working directory.
        default_model: LLM model name recorded in ``project.yaml`` for
            later ``--llm`` runs from the CLI.

    Returns:
        ``{"ok": True, "path": "<abs path>", "name": "<name>"}`` on
        success, or ``{"ok": False, "error": ..., "message": ...}`` if
        a project already exists at the target.
    """
    try:
        project = ProjectContext.init(
            name,
            parent_dir=parent_dir,
            default_model=default_model,
        )
    except ProjectExistsError as e:
        return _error(e)

    # v0.7.2: stamp the init itself into the decisions log so the
    # ledger has a creation marker right from the first MCP call.
    # All ledger writes are best-effort — init itself already succeeded.
    with contextlib.suppress(Exception):
        project.append_decision(
            command="init",
            summary=f"Project '{name}' initialised via MCP",
            reasoning=f"parent_dir={parent_dir}; via=mcp",
        )
        from datetime import datetime, timezone

        with (project.path / "advice.log").open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "command": "init",
                        "name": name,
                        "via": "mcp",
                    }
                )
                + "\n"
            )

    return {"ok": True, "path": str(project.path), "name": name}


@mcp.tool()
def mlcompass_status(
    project_path: str = ".",
    recent_decisions: int = 5,
) -> dict[str, Any]:
    """Summarise the active mlcompass project context.

    Walks up from ``project_path`` looking for ``.mlcompass/`` (mirrors
    how ``git`` discovers a repository), then assembles a JSON-friendly
    snapshot.

    Args:
        project_path: Path inside or above an mlcompass project.
        recent_decisions: How many of the latest decision entries to
            include.

    Returns:
        ``{"ok": True, "project", "state", "command_counts",
        "decisions", "total_decisions"}`` on success. The ``project``
        block is the static ``project.yaml`` metadata; ``state`` mirrors
        ``context.json``'s active fields; ``command_counts`` aggregates
        ``advice.log`` per command name.
    """
    try:
        project = ProjectContext.load(search_from=project_path)
    except ProjectNotFoundError as e:
        return _error(e)

    state = project.read_context()
    decisions = state.get("decisions") or []

    counts: Counter[str] = Counter()
    log_path = project.path / "advice.log"
    if log_path.is_file():
        for raw in log_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            command = entry.get("command")
            if isinstance(command, str):
                counts[command] += 1

    return {
        "ok": True,
        "project": project.project_meta,
        "state": {
            "project_type": state.get("project_type"),
            "target_column": state.get("target_column"),
            "active_dataset": state.get("active_dataset"),
            "current_run": state.get("current_run"),
            "preferred_models": state.get("preferred_models") or [],
        },
        "command_counts": dict(counts),
        "decisions": decisions[-recent_decisions:] if decisions else [],
        "total_decisions": len(decisions),
    }


# --------------------------------------------------------------------------- #
# Pipeline tools                                                              #
# --------------------------------------------------------------------------- #


@mcp.tool()
def mlcompass_advise(
    dataset_path: str,
    target: str | None = None,
    sample_rows: int | None = None,
) -> dict[str, Any]:
    """Analyse a dataset and surface schema, target hint, task hint, warnings.

    Args:
        dataset_path: Path to a CSV / Parquet / Excel / JSON / JSONL file.
        target: Known target column name. Auto-detected if omitted.
        sample_rows: Read only the first ``N`` rows (useful for very
            large files where a full pass is wasteful).

    Returns:
        ``{"path", "format", "shape", "columns", "target_hint",
        "task_hint", "warnings"}``. Use the ``warnings`` list and the
        column-level breakdown to reason about feature engineering and
        model choice.
    """
    try:
        result = analyze_dataset(
            dataset_path,
            target_column=target,
            sample_rows=sample_rows,
        )
    except (ValueError, FileNotFoundError, OSError) as e:
        return _error(e)

    # v0.7.2: persist to project ledger so `mlcompass_status` reflects
    # the MCP-driven activity the same way it does CLI activity.
    target_hint = result.get("target_hint") or {}
    task_hint = result.get("task_hint") or {}
    shape = result.get("shape") or {}
    summary = (
        f"Advise: {task_hint.get('type', 'unknown')} "
        f"target={target_hint.get('column', 'auto')} "
        f"({shape.get('rows', '?')}×{shape.get('cols', '?')})"
    )
    _persist_to_ledger(
        search_from=_path_dir(dataset_path),
        command="advise",
        summary=summary,
        reasoning=f"dataset={dataset_path}; via=mcp",
        log_extra={
            "dataset": dataset_path,
            "target": target_hint.get("column"),
            "task": task_hint.get("type"),
            "via": "mcp",
        },
    )
    return cast(dict[str, Any], _json_safe(result))


@mcp.tool()
def mlcompass_audit(
    script_path: str,
    skip_rules: list[str] | None = None,
) -> dict[str, Any]:
    """Statically analyse a Python training script for common ML mistakes.

    Eight pure-AST rules: ``seed``, ``val_split``, ``optimizer``,
    ``loss_stability``, ``dataloader``, ``grad_clipping``, ``eval_mode``,
    ``batch_size``.

    Args:
        script_path: Path to the Python source file.
        skip_rules: Rule IDs to skip on this run.

    Returns:
        ``{"path", "lines", "frameworks", "findings"}``. Each finding
        carries ``rule_id``, ``severity`` (error / warning / info),
        ``message``, ``suggestion``, and ``line``.
    """
    try:
        skip = set(skip_rules) if skip_rules else None
        result = audit_script(script_path, skip_rules=skip)
    except (SyntaxError, FileNotFoundError, OSError) as e:
        return _error(e)

    findings = result.get("findings") or []
    counts = Counter(f.get("severity", "info") for f in findings)
    summary = (
        f"Audit: {counts.get('error', 0)} error, "
        f"{counts.get('warning', 0)} warning, "
        f"{counts.get('info', 0)} info"
    )
    _persist_to_ledger(
        search_from=_path_dir(script_path),
        command="audit",
        summary=summary,
        reasoning=f"script={script_path}; via=mcp",
        log_extra={"script": script_path, "findings": len(findings), "via": "mcp"},
    )
    return result


@mcp.tool()
def mlcompass_watch(log_path: str) -> dict[str, Any]:
    """Scan a training log / TensorBoard run / W&B run for anomalies.

    Source type is auto-detected from the path: plain-text logs,
    ``events.out.tfevents.*`` files (TensorBoard), and W&B local run
    directories are all supported. Streaming (``--follow``) is a CLI-only
    feature; the MCP tool is one-shot.

    Args:
        log_path: Path to the metric source.

    Returns:
        ``{"ok", "source", "snapshot_count", "last_epoch", "metrics",
        "findings"}``. Detectors: ``nan``, ``divergence``, ``plateau``,
        ``overfitting``. NaN / Inf metric values are mapped to ``null``
        for JSON safety; the ``nan`` detector still surfaces them in the
        findings list.
    """
    try:
        source, snapshots = load_snapshots(log_path)
    except (FileNotFoundError, ValueError, OSError) as e:
        return _error(e)

    snapshots = merge_consecutive_same_epoch(snapshots)
    findings = run_all_detectors(snapshots)

    last_epoch: int | None = None
    for snap in reversed(snapshots):
        if snap.epoch is not None:
            last_epoch = snap.epoch
            break

    counts = Counter(f.to_dict().get("severity", "info") for f in findings)
    summary = (
        f"Watch ({source}): {len(findings)} finding(s) "
        f"[{counts.get('error', 0)}e/{counts.get('warning', 0)}w]"
    )
    _persist_to_ledger(
        search_from=_path_dir(log_path),
        command="watch",
        summary=summary,
        reasoning=f"log={log_path}; source={source}; via=mcp",
        log_extra={
            "log": log_path,
            "source": source,
            "snapshots": len(snapshots),
            "findings": len(findings),
            "via": "mcp",
        },
    )

    return {
        "ok": True,
        "source": source,
        "snapshot_count": len(snapshots),
        "last_epoch": last_epoch,
        "metrics": [_json_safe({"epoch": s.epoch, "step": s.step, **s.metrics}) for s in snapshots],
        "findings": [f.to_dict() for f in findings],
    }


@mcp.tool()
def mlcompass_compare(
    run_a: str,
    run_b: str,
    project_path: str | None = None,
) -> dict[str, Any]:
    """Side-by-side comparison of two training runs.

    Each run identifier may be either an absolute / relative path to a
    run directory, or a run name resolved under ``<project>/runs/<name>``.
    Pass ``project_path`` so name-based lookups succeed.

    Args:
        run_a: First run identifier.
        run_b: Second run identifier.
        project_path: Path inside an mlcompass project. Optional when
            the run identifiers are full paths.

    Returns:
        Structured comparison with config diff, per-metric winner, and
        a final verdict (``A wins / B wins / mixed / tie``).
    """
    project = None
    if project_path is not None:
        try:
            project = ProjectContext.load(search_from=project_path)
        except ProjectNotFoundError:
            project = None

    try:
        a = load_run(run_a, project=project)
        b = load_run(run_b, project=project)
    except RunNotFoundError as e:
        return _error(e)

    comparison = compare_runs(a, b)
    verdict = comparison.get("verdict")
    if isinstance(verdict, dict):
        verdict_label = verdict.get("label") or verdict.get("verdict") or "mixed"
    else:
        verdict_label = str(verdict) if verdict else "mixed"
    summary = f"Compare: {run_a} vs {run_b} ⇒ {verdict_label}"
    _persist_to_ledger(
        search_from=project_path or ".",
        command="compare",
        summary=summary,
        reasoning=f"run_a={run_a}; run_b={run_b}; via=mcp",
        log_extra={"run_a": run_a, "run_b": run_b, "via": "mcp"},
    )
    return cast(dict[str, Any], _json_safe(comparison))


@mcp.tool()
def mlcompass_evaluate(
    results_path: str,
    y_true_col: str | None = None,
    y_pred_col: str | None = None,
    y_prob_col: str | None = None,
    task: str | None = None,
    hard_examples_k: int = 5,
) -> dict[str, Any]:
    """Post-training evaluation on a predictions table.

    Detects task type from the columns when possible (binary /
    multiclass classification or regression). Includes a leakage-smell
    warning when the metrics look implausibly perfect — AUC > 0.995,
    accuracy > 0.99, or R² > 0.999 over at least 50 rows.

    Args:
        results_path: Path to a predictions file (CSV / Parquet / Excel
            / JSON / JSONL).
        y_true_col / y_pred_col / y_prob_col: Override auto-detection
            (case-insensitive name match).
        task: Force the task type. Valid: ``binary_classification``,
            ``multiclass_classification``, ``regression``.
        hard_examples_k: How many worst-error rows to surface.

    Returns:
        Metrics + confusion matrix + (binary) threshold sweep + (multi-
        class) per-class breakdown + (regression) residual summary +
        hard examples + warnings.
    """
    try:
        df = load_results(results_path)
        result = run_evaluation(
            df,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            y_prob_col=y_prob_col,
            task=task,
            hard_examples_k=hard_examples_k,
        )
    except (EvaluationError, FileNotFoundError, OSError) as e:
        return _error(e)

    metrics = result.get("metrics") or {}
    inferred_task = result.get("task", "unknown")
    leakage = result.get("leakage_investigation")
    headline = ", ".join(f"{k}={v}" for k, v in list(metrics.items())[:3]) or "no metrics"
    smell = " [LEAKAGE-SMELL]" if leakage else ""
    summary = f"Evaluate ({inferred_task}): {headline}{smell}"
    _persist_to_ledger(
        search_from=_path_dir(results_path),
        command="evaluate",
        summary=summary,
        reasoning=f"results={results_path}; via=mcp",
        log_extra={
            "results": results_path,
            "task": inferred_task,
            "metrics": metrics,
            "leakage_smell": bool(leakage),
            "via": "mcp",
        },
    )
    return cast(dict[str, Any], _json_safe(result))


@mcp.tool()
def mlcompass_deploy(
    model_path: str,
    requirements_path: str | None = None,
    target: str = "local",
) -> dict[str, Any]:
    """Deployment-readiness check on a trained model file.

    The model is **not** loaded — only file metadata, magic bytes, and
    (optionally) the dependency manifest are inspected. This is by
    design: loading arbitrary pickles / GPU-bound checkpoints inside an
    advisor would be a footgun.

    Args:
        model_path: Path to the trained model file (``.pt``, ``.pkl``,
            ``.joblib``, ``.h5``, ``.onnx``, ``.safetensors``, …).
        requirements_path: Optional path to ``requirements.txt``,
            ``pyproject.toml``, or ``environment.yml``.
        target: One of ``local``, ``sagemaker``, ``lambda``,
            ``kubernetes``, ``vertex``.

    Returns:
        ``{"model", "dependencies", "target", "checklist", "warnings"}``.
        Target-specific findings (e.g. Lambda's 250 MB ceiling) appear
        inside ``warnings``.
    """
    try:
        result = assess_deployment(
            model_path,
            requirements_path=requirements_path,
            target=target,
        )
    except (DeployAnalysisError, FileNotFoundError, OSError) as e:
        return _error(e)

    model_info = result.get("model") or {}
    warnings = result.get("warnings") or []
    summary = (
        f"Deploy ({target}): {model_info.get('format', 'unknown')} "
        f"{model_info.get('size_class', '')}, "
        f"{len(warnings)} warning(s)"
    )
    _persist_to_ledger(
        search_from=_path_dir(model_path),
        command="deploy",
        summary=summary,
        reasoning=f"model={model_path}; target={target}; via=mcp",
        log_extra={
            "model": model_path,
            "target": target,
            "warnings": len(warnings),
            "via": "mcp",
        },
    )
    return cast(dict[str, Any], _json_safe(result))


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main() -> None:
    """Stdio MCP server entry point — registered as ``mlcompass-mcp``."""
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
