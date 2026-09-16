"""Project context: persistent state stored in ``.mlcompass/``.

The directory layout is described in ``ARCHITECTURE.md §2``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml
from filelock import FileLock, Timeout

from . import __version__

DEFAULT_PROJECT_DIR = ".mlcompass"


# --------------------------------------------------------------------------- #
# Durable writes                                                              #
# --------------------------------------------------------------------------- #
#
# ``context.json`` is updated by read-modify-write over the whole file.
# Two problems that used to be live:
#
# 1. No serialisation. Two writers interleave as read-A / read-B /
#    write-A / write-B and B silently drops A's entry. Measured: 80
#    concurrent appends left 40 decisions on disk.
# 2. No atomicity. ``Path.write_text`` truncates the file and *then*
#    fills it, so any concurrent reader landing in that window sees an
#    empty or partial file. Measured: a reader observed a torn file
#    mid-write.
#
# The fix is the standard pair: one lock per file for writers in this
# process, and write-temp-then-``os.replace`` so every state a reader
# can observe is either the whole old file or the whole new one.
#
# Two locks, because there are two kinds of writer.
#
# The threading lock serialises writers inside one interpreter, which is
# where the demonstrated loss happened: the agent and its tools share a
# process.
#
# The file lock serialises writers across processes, which is the case
# the threading lock cannot see. It is not hypothetical here. mlcompass
# ships a CLI and an MCP server over the same ``.mlcompass/`` directory,
# and an editor session running the MCP server while the developer runs
# a CLI command in a terminal is the ordinary way to use the tool, not
# an unusual one. Without it, read-A / read-B / write-A / write-B
# silently drops A's decision from the audit ledger — and an audit
# ledger that loses entries under normal use is worse than no ledger,
# because it looks complete.
#
# Order matters and is not arbitrary: acquire the thread lock first,
# then the file lock. The reverse order lets two threads in one process
# both block on the file lock while holding nothing, and the reentrancy
# the thread lock provides (``read_context`` is called under it) is lost.
#
# ``filelock`` is a dependency rather than an optional import. A lock
# that silently does nothing when a package is missing is worse than no
# lock: it moves a visible failure to a silent one, which is the
# specific trade this module exists to refuse.

_LOCKS: dict[str, threading.RLock] = {}
_FILE_LOCKS: dict[str, FileLock] = {}
_LOCKS_GUARD = threading.Lock()

# Windows refuses os.replace while another handle has the destination
# open. That window is microseconds wide; a short retry closes it.
_REPLACE_ATTEMPTS = 20
_REPLACE_BACKOFF_S = 0.005

# Long enough that a slow write on a loaded machine is not mistaken for a
# stuck process, short enough that a genuinely stale lock surfaces as an
# error the same minute rather than as a hang.
_FILE_LOCK_TIMEOUT_S = 30.0


def _lock_for(path: Path) -> threading.RLock:
    """One reentrant lock per file, shared across all ProjectContext instances."""
    key = str(path.resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock
        return lock


def _file_lock_for(path: Path) -> FileLock:
    """One reentrant cross-process lock per file, shared in this process."""
    key = str(path.resolve())
    with _LOCKS_GUARD:
        lock = _FILE_LOCKS.get(key)
        if lock is None:
            lock = FileLock(str(path.with_name(path.name + ".lock")),
                            timeout=_FILE_LOCK_TIMEOUT_S)
            _FILE_LOCKS[key] = lock
        return lock


@contextmanager
def _exclusive(path: Path) -> Iterator[None]:
    """Hold both locks for ``path``: in-process, then cross-process.

    The lock file sits beside the target with a ``.lock`` suffix and is
    left in place afterwards. Deleting it would reintroduce the race it
    prevents, because a second process can create and acquire a fresh
    lock file in the window between the first process unlinking it and
    releasing it.

    One ``FileLock`` instance per path, shared, because this must be
    reentrant. ``append_decision`` calls ``read_context`` while holding
    the lock, and a fresh ``FileLock`` on an already-locked path blocks
    against its own process -- a self-deadlock that surfaces as a
    30-second hang and then a timeout claiming another process is
    responsible. A shared instance nests via its own counter, and the
    threading lock above guarantees only one thread is ever inside.

    A timeout rather than an indefinite wait: a stale lock from a process
    that died holding it would otherwise hang every later command with no
    explanation. The error says which file and what to do.
    """
    with _lock_for(path):
        if not path.parent.is_dir():
            # No project directory, so no file to race over and nowhere to put
            # a lock file. Yield under the thread lock alone and let the write
            # itself raise, which gives the caller the error it had before this
            # lock existed rather than a FileNotFoundError about a .lock file
            # nobody asked for.
            yield
            return
        file_lock = _file_lock_for(path)
        try:
            file_lock.acquire()
        except Timeout as exc:
            raise TimeoutError(
                f"could not lock {path} within {_FILE_LOCK_TIMEOUT_S}s: another "
                f"mlcompass process is holding {lock_path.name}. If no other "
                f"process is running, delete that file and retry."
            ) from exc
        try:
            yield
        finally:
            file_lock.release()


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` in one indivisible step.

    The temp file is created in the destination directory so the rename
    stays on one filesystem, which is what makes it atomic.
    """
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())

        last: OSError | None = None
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp_name, path)
                return
            except PermissionError as e:  # pragma: no cover — Windows-only window
                last = e
                time.sleep(_REPLACE_BACKOFF_S * (attempt + 1))
        raise last if last is not None else OSError(f"could not replace {path}")
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


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

    @property
    def _context_path(self) -> Path:
        return self.path / "context.json"

    def read_context(self) -> dict[str, Any]:
        """Read the dynamic context (``context.json``)."""
        with _exclusive(self._context_path):
            data: dict[str, Any] = json.loads(self._context_path.read_text(encoding="utf-8"))
        return data

    def write_context(self, updates: dict[str, Any]) -> None:
        """Merge ``updates`` into ``context.json``.

        Top-level keys are replaced. For lists like ``decisions`` prefer
        ``append_decision`` so timestamps are added automatically.

        The read and the write happen under one lock: without that, a
        concurrent writer's update lands between them and is lost when
        this one writes back its stale copy.
        """
        with _exclusive(self._context_path):
            current = self.read_context()
            current.update(updates)
            _atomic_write_text(self._context_path, json.dumps(current, indent=2))

    def append_decision(
        self,
        command: str,
        summary: str,
        *,
        reasoning: str = "",
    ) -> None:
        """Append a timestamped decision entry to ``decisions``."""
        with _exclusive(self._context_path):
            ctx = self.read_context()
            ctx.setdefault("decisions", []).append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "command": command,
                    "summary": summary,
                    "reasoning": reasoning,
                }
            )
            _atomic_write_text(self._context_path, json.dumps(ctx, indent=2))

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
        record_path = self.path / "datasets" / f"{fingerprint}.json"
        with _lock_for(record_path):
            _atomic_write_text(record_path, json.dumps(record, indent=2))
        return fingerprint

    @staticmethod
    def _fingerprint(p: Path) -> str:
        """Stable per-file fingerprint based on path + mtime."""
        marker = f"{p.resolve()}::{p.stat().st_mtime_ns}".encode()
        return hashlib.sha256(marker).hexdigest()[:16]
