"""Project context: persistent state stored in ``.mlcompass/``.

The directory layout is described in ``ARCHITECTURE.md §2``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import __version__

DEFAULT_PROJECT_DIR = ".mlcompass"


class ProjectExistsError(FileExistsError):
    """Raised when init is called but ``.mlcompass/`` already exists."""


class ProjectNotFoundError(FileNotFoundError):
    """Raised when load is called but no ``.mlcompass/`` is found."""


@dataclass
class ProjectContext:
    """Wraps a single ``.mlcompass/`` project directory.

    Layout::

        .mlcompass/
        ├── project.yaml      # static metadata
        ├── context.json      # dynamic state (decisions, recommendations)
        ├── datasets/         # registered datasets
        ├── runs/             # training run history
        ├── advice.log        # advisor recommendation history
        └── cache/            # tool result + LLM prompt cache
    """

    path: Path  # the ``.mlcompass/`` directory itself

    # ---------------------- Construction ----------------------

    @classmethod
    def init(
        cls,
        name: str,
        parent_dir: Path | str = ".",
        *,
        default_model: str = "claude-opus-4-7",
    ) -> ProjectContext:
        """Create a new ``.mlcompass/`` directory under ``parent_dir``.

        Args:
            name: Human-readable project name (e.g., ``"churn-model"``).
            parent_dir: Directory in which ``.mlcompass/`` will be created.
            default_model: Default LLM model name for the project.

        Returns:
            The newly created ``ProjectContext``.

        Raises:
            ProjectExistsError: If a project already exists at the target.
        """
        parent_dir = Path(parent_dir).resolve()
        target = parent_dir / DEFAULT_PROJECT_DIR

        if target.exists():
            raise ProjectExistsError(f"Project already exists at {target}")

        # Directory tree
        target.mkdir(parents=True)
        (target / "datasets").mkdir()
        (target / "runs").mkdir()
        (target / "cache").mkdir()

        # Static metadata
        project_meta = {
            "name": name,
            "created": datetime.now(timezone.utc).isoformat(),
            "mlcompass_version": __version__,
            "default_model": default_model,
        }
        (target / "project.yaml").write_text(
            yaml.safe_dump(project_meta, sort_keys=False),
            encoding="utf-8",
        )

        # Dynamic context starts empty
        initial_context: dict[str, Any] = {
            "project_type": None,
            "target_column": None,
            "preferred_models": [],
            "active_dataset": None,
            "current_run": None,
            "decisions": [],
        }
        (target / "context.json").write_text(
            json.dumps(initial_context, indent=2),
            encoding="utf-8",
        )

        # Advice log starts empty
        (target / "advice.log").touch()

        # Local .gitignore so cache/ and runs/ don't pollute the user's repo
        (target / ".gitignore").write_text(
            "cache/\nruns/\n*.pyc\n__pycache__/\n",
            encoding="utf-8",
        )

        return cls(path=target)

    @classmethod
    def load(cls, search_from: Path | str = ".") -> ProjectContext:
        """Find and load an existing project by walking up from ``search_from``.

        Mirrors the behaviour of ``git`` when discovering a repository.

        Raises:
            ProjectNotFoundError: If no ``.mlcompass/`` is found between
                ``search_from`` and the filesystem root.
        """
        current = Path(search_from).resolve()
        for candidate in [current, *current.parents]:
            target = candidate / DEFAULT_PROJECT_DIR
            if target.is_dir():
                return cls(path=target)
        raise ProjectNotFoundError(
            f"No {DEFAULT_PROJECT_DIR}/ found at or above {current}. "
            "Run `mlcompass init <name>` to create one."
        )

    # ---------------------- Read / write ----------------------

    @property
    def project_meta(self) -> dict[str, Any]:
        """Static metadata loaded from ``project.yaml``."""
        data: dict[str, Any] = yaml.safe_load(
            (self.path / "project.yaml").read_text(encoding="utf-8")
        )
        return data

    def read_context(self) -> dict[str, Any]:
        """Read the dynamic context (``context.json``)."""
        data: dict[str, Any] = json.loads((self.path / "context.json").read_text(encoding="utf-8"))
        return data

    def write_context(self, updates: dict[str, Any]) -> None:
        """Merge ``updates`` into ``context.json``.

        Top-level keys are replaced. For lists like ``decisions`` prefer
        ``append_decision`` so timestamps are added automatically.
        """
        current = self.read_context()
        current.update(updates)
        (self.path / "context.json").write_text(
            json.dumps(current, indent=2),
            encoding="utf-8",
        )

    def append_decision(
        self,
        command: str,
        summary: str,
        *,
        reasoning: str = "",
    ) -> None:
        """Append a timestamped decision entry to ``decisions``."""
        ctx = self.read_context()
        ctx.setdefault("decisions", []).append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "command": command,
                "summary": summary,
                "reasoning": reasoning,
            }
        )
        (self.path / "context.json").write_text(
            json.dumps(ctx, indent=2),
            encoding="utf-8",
        )

    # ---------------------- Dataset registry ----------------------

    def register_dataset(
        self,
        dataset_path: Path | str,
        meta: dict[str, Any],
    ) -> str:
        """Save dataset metadata under ``datasets/<fingerprint>.json``.

        The fingerprint is the truncated SHA-256 of the absolute path plus
        the file's modification time, so re-registering an unchanged file
        is idempotent.

        Returns:
            The fingerprint (16-character hex).
        """
        dataset_path = Path(dataset_path).resolve()
        fingerprint = self._fingerprint(dataset_path)
        record = {
            "path": str(dataset_path),
            "registered_at": datetime.now(timezone.utc).isoformat(),
            **meta,
        }
        (self.path / "datasets" / f"{fingerprint}.json").write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )
        return fingerprint

    @staticmethod
    def _fingerprint(p: Path) -> str:
        """Stable per-file fingerprint based on path + mtime."""
        marker = f"{p.resolve()}::{p.stat().st_mtime_ns}".encode()
        return hashlib.sha256(marker).hexdigest()[:16]
