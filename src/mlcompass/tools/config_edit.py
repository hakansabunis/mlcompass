"""Permission-gated config-file edits.

Used by ``mlcompass watch --apply`` to take the diagnostician's
``suggested_edits`` and, after asking the user once per edit, modify
the user's training-config file in place. A timestamped ``.bak``
backup is written before the first applied edit so the original file
is always recoverable.

The module is deliberately small and free of any LLM dependency — the
agent already produced the structured edits; this layer just turns
them into careful filesystem changes.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

# --------------------------------------------------------------------------- #
# Public types                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class ConfigEdit:
    """A single proposed edit, as the agent describes it."""

    key: str
    current_value: Any
    proposed_value: Any
    rationale: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConfigEdit":
        return cls(
            key=str(payload["key"]),
            current_value=payload.get("current_value"),
            proposed_value=payload.get("proposed_value"),
            rationale=str(payload.get("rationale", "")),
        )


@dataclass
class ApplyResult:
    """Aggregated outcome of a permission-gated edit session."""

    applied: list[ConfigEdit] = field(default_factory=list)
    skipped: list[ConfigEdit] = field(default_factory=list)
    rejected: list[ConfigEdit] = field(default_factory=list)
    backup_path: Path | None = None

    @property
    def changed_count(self) -> int:
        return len(self.applied)


class ConfigEditError(Exception):
    """Raised on filesystem / format failures during an edit session."""


ConfirmFn = Callable[[ConfigEdit, dict[str, Any]], bool]
"""``(edit, current_config_dict) -> bool``.

The callback is asked once per proposed edit. The current config
state is passed so terminal renderers can show "this is the value
we're about to overwrite".
"""


# --------------------------------------------------------------------------- #
# Loaders / writers                                                           #
# --------------------------------------------------------------------------- #


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML or JSON config file as a top-level mapping."""
    path = Path(path)
    if not path.is_file():
        raise ConfigEditError(f"Config file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    try:
        if suffix == ".json":
            data = json.loads(raw)
        else:
            data = yaml.safe_load(raw)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigEditError(f"Could not parse config: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigEditError(
            f"Config root must be a mapping; got {type(data).__name__} in {path}."
        )
    return data


def write_config(path: Path, data: dict[str, Any]) -> None:
    """Write a config back to disk in the same format the path implies."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    else:
        path.write_text(
            yaml.safe_dump(data, sort_keys=False),
            encoding="utf-8",
        )


# --------------------------------------------------------------------------- #
# Edit driver                                                                 #
# --------------------------------------------------------------------------- #


def _set_dotted(data: dict[str, Any], key: str, value: Any) -> None:
    """Set a dotted ``a.b.c`` path inside ``data``, creating dicts on demand."""
    parts = key.split(".")
    parent: dict[str, Any] = data
    for part in parts[:-1]:
        nxt = parent.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            parent[part] = nxt
        parent = nxt
    parent[parts[-1]] = value


def _get_dotted(data: dict[str, Any], key: str) -> tuple[bool, Any]:
    """Return ``(found, value)`` for a dotted lookup."""
    parts = key.split(".")
    current: Any = data
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def apply_edits(
    config_path: Path | str,
    edits: list[ConfigEdit | dict[str, Any]],
    *,
    confirm_fn: ConfirmFn,
) -> ApplyResult:
    """Walk ``edits`` and apply each one the user confirms.

    For every edit:
    - skip silently if the live config doesn't actually carry the key
      (the diagnostician may have proposed a key that lives in a
      different file)
    - skip silently if the live value already equals the proposed value
    - otherwise call ``confirm_fn`` and either apply or record as
      rejected

    A timestamped ``.bak`` is written before the first applied edit.
    """
    config_path = Path(config_path)
    data = load_config(config_path)

    result = ApplyResult()
    backup_written = False

    for raw in edits:
        edit = (
            raw if isinstance(raw, ConfigEdit) else ConfigEdit.from_dict(raw)
        )

        found, live_value = _get_dotted(data, edit.key)
        if not found:
            result.skipped.append(edit)
            continue
        if live_value == edit.proposed_value:
            result.skipped.append(edit)
            continue

        # Surface the live value in case the agent's `current_value`
        # was stale; downstream UI can show both side by side.
        snapshot_edit = ConfigEdit(
            key=edit.key,
            current_value=live_value,
            proposed_value=edit.proposed_value,
            rationale=edit.rationale,
        )

        if not confirm_fn(snapshot_edit, dict(data)):
            result.rejected.append(snapshot_edit)
            continue

        if not backup_written:
            result.backup_path = _write_backup(config_path)
            backup_written = True

        _set_dotted(data, edit.key, edit.proposed_value)
        result.applied.append(snapshot_edit)

    if result.applied:
        write_config(config_path, data)

    return result


def _write_backup(config_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = config_path.with_suffix(config_path.suffix + f".{timestamp}.bak")
    shutil.copy2(config_path, backup)
    return backup
