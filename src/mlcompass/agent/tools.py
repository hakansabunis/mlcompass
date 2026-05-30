"""Shared tool registry consumed by every agent backend.

This is the single source of truth for the eight tools the agent can
call. Each tool has:

- a hand-written JSON Schema (so the LLM gets rich argument hints), and
- a Python dispatcher that runs the work and returns a JSON-safe dict.

Dispatchers reuse the wrappers from :mod:`mlcompass.mcp_server` so the
CLI agent, the MCP server, and the (future) Agent-SDK backend stay in
lockstep — fix a tool once, all three layers see the fix.

The registry intentionally classifies each tool as ``mutates`` or not.
The orchestrator's permission gate consults this flag to decide whether
to ask the user before dispatch (mutating tools require confirmation
unless ``--auto-approve`` is set).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..mcp_server import (
    mlcompass_advise,
    mlcompass_audit,
    mlcompass_compare,
    mlcompass_deploy,
    mlcompass_evaluate,
    mlcompass_init,
    mlcompass_status,
    mlcompass_watch,
)


@dataclass(frozen=True)
class ToolSpec:
    """One agent-callable tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    dispatcher: Callable[..., dict[str, Any]]
    mutates: bool = False  # True ⇒ permission gate fires unless auto-approved.


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #


_DATASET_PATH = {
    "type": "string",
    "description": "Filesystem path to the dataset (CSV, Parquet, Excel, JSON, JSONL).",
}
_SCRIPT_PATH = {
    "type": "string",
    "description": "Filesystem path to the Python training script.",
}
_LOG_PATH = {
    "type": "string",
    "description": (
        "Filesystem path to a training metric source: a plain-text log, "
        "a TensorBoard event file or directory, or a W&B local run dir."
    ),
}
_PROJECT_PATH = {
    "type": "string",
    "description": (
        "Path inside or above an mlcompass project. Walks up looking for a .mlcompass/ directory."
    ),
    "default": ".",
}


TOOL_REGISTRY: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="mlcompass_init",
        description=(
            "Create a new .mlcompass/ project directory. Use this when "
            "the user starts a new ML project from scratch or no "
            ".mlcompass/ exists yet. Side-effect: writes files to disk."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Human-readable project name (e.g. 'churn-model').",
                },
                "parent_dir": {
                    "type": "string",
                    "description": "Directory under which .mlcompass/ is created.",
                    "default": ".",
                },
                "default_model": {
                    "type": "string",
                    "description": "Default LLM model name recorded in project.yaml.",
                    "default": "claude-opus-4-7",
                },
            },
            "required": ["name"],
        },
        dispatcher=mlcompass_init,
        mutates=True,
    ),
    ToolSpec(
        name="mlcompass_status",
        description=(
            "Summarise the active mlcompass project: name, active dataset, "
            "target column, preferred models, per-command activity counts, "
            "and the latest decisions. Read-only."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_path": _PROJECT_PATH,
                "recent_decisions": {
                    "type": "integer",
                    "description": "How many of the latest decision entries to include.",
                    "default": 5,
                },
            },
        },
        dispatcher=mlcompass_status,
    ),
    ToolSpec(
        name="mlcompass_advise",
        description=(
            "Analyse a dataset and recommend models, features, and "
            "pitfalls. Returns schema, target hint, task hint, and "
            "warnings. The first tool to reach for when the user shares "
            "raw data."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "dataset_path": _DATASET_PATH,
                "target": {
                    "type": "string",
                    "description": "Known target column name. Auto-detected if omitted.",
                },
                "sample_rows": {
                    "type": "integer",
                    "description": (
                        "Read only the first N rows (use for files >10 GB "
                        "where a full pass is wasteful)."
                    ),
                },
            },
            "required": ["dataset_path"],
        },
        dispatcher=mlcompass_advise,
    ),
    ToolSpec(
        name="mlcompass_audit",
        description=(
            "Statically analyse a Python training script for the eight "
            "most common ML mistakes (seed, val_split, optimizer, "
            "loss_stability, dataloader, grad_clipping, eval_mode, "
            "batch_size). Pure AST — no execution."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "script_path": _SCRIPT_PATH,
                "skip_rules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Rule IDs to skip on this run.",
                },
            },
            "required": ["script_path"],
        },
        dispatcher=mlcompass_audit,
    ),
    ToolSpec(
        name="mlcompass_watch",
        description=(
            "Scan a training log / TensorBoard run / W&B run for "
            "anomalies. Source type is auto-detected from the path. "
            "Detectors: nan, divergence, plateau, overfitting."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "log_path": _LOG_PATH,
            },
            "required": ["log_path"],
        },
        dispatcher=mlcompass_watch,
    ),
    ToolSpec(
        name="mlcompass_compare",
        description=(
            "Side-by-side comparison of two training runs: config diff, "
            "final-metric winner, and an overall verdict (A wins / B "
            "wins / mixed / tie). Each run identifier is either a path "
            "to a run directory or a run name under <project>/runs/<name>."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "run_a": {"type": "string", "description": "First run identifier."},
                "run_b": {"type": "string", "description": "Second run identifier."},
                "project_path": _PROJECT_PATH,
            },
            "required": ["run_a", "run_b"],
        },
        dispatcher=mlcompass_compare,
    ),
    ToolSpec(
        name="mlcompass_evaluate",
        description=(
            "Post-training evaluation on a predictions table. Computes "
            "metrics + confusion matrix + (binary) threshold sweep + "
            "(multiclass) per-class breakdown + (regression) residuals. "
            "Includes a leakage-smell warning when results look "
            "implausibly perfect (AUC > 0.995, acc > 0.99, R² > 0.999 "
            "over at least 50 rows)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "results_path": {
                    "type": "string",
                    "description": (
                        "Path to a predictions file (CSV / Parquet / Excel / JSON / JSONL)."
                    ),
                },
                "y_true_col": {"type": "string"},
                "y_pred_col": {"type": "string"},
                "y_prob_col": {"type": "string"},
                "task": {
                    "type": "string",
                    "enum": [
                        "binary_classification",
                        "multiclass_classification",
                        "regression",
                    ],
                    "description": "Force the task type. Inferred from columns if omitted.",
                },
                "hard_examples_k": {
                    "type": "integer",
                    "description": "How many worst-error rows to surface.",
                    "default": 5,
                },
            },
            "required": ["results_path"],
        },
        dispatcher=mlcompass_evaluate,
    ),
    ToolSpec(
        name="mlcompass_deploy",
        description=(
            "Deployment-readiness check on a trained model file. The "
            "model is NOT loaded — only metadata, magic bytes, and the "
            "(optional) dependency manifest are inspected. Targets: "
            "local, sagemaker, lambda, kubernetes, vertex."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "model_path": {
                    "type": "string",
                    "description": "Path to the trained model file.",
                },
                "requirements_path": {
                    "type": "string",
                    "description": (
                        "Optional path to requirements.txt / pyproject.toml / environment.yml."
                    ),
                },
                "target": {
                    "type": "string",
                    "enum": ["local", "sagemaker", "lambda", "kubernetes", "vertex"],
                    "default": "local",
                },
            },
            "required": ["model_path"],
        },
        dispatcher=mlcompass_deploy,
    ),
)


# --------------------------------------------------------------------------- #
# Lookup + dispatch                                                           #
# --------------------------------------------------------------------------- #


BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_REGISTRY}


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call by name, returning a JSON-safe dict.

    Falls through to a uniform error envelope on unknown tool name or
    bad argument shape, matching the MCP layer's convention so the
    calling LLM sees consistent failure signals.
    """
    spec = BY_NAME.get(name)
    if spec is None:
        return {
            "ok": False,
            "error": "UnknownTool",
            "message": (f"No tool named {name!r}. Available: {', '.join(sorted(BY_NAME))}"),
        }
    try:
        return spec.dispatcher(**arguments)
    except TypeError as e:
        return {"ok": False, "error": "BadArguments", "message": str(e)}


def anthropic_tool_specs() -> list[dict[str, Any]]:
    """Return the registry in the shape Anthropic's ``messages.create`` expects.

    ``messages.create`` wants ``[{"name", "description", "input_schema"}]``.
    The dispatcher is private to mlcompass — Anthropic doesn't see it.
    """
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
        }
        for spec in TOOL_REGISTRY
    ]
