"""Tests for the permission-gated config editor (``tools.config_edit``)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mlcompass.tools.config_edit import (
    ConfigEdit,
    ConfigEditError,
    apply_edits,
    load_config,
    write_config,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _yes(_edit: ConfigEdit, _config: dict) -> bool:
    return True


def _no(_edit: ConfigEdit, _config: dict) -> bool:
    return False


# --------------------------------------------------------------------------- #
# load_config / write_config                                                  #
# --------------------------------------------------------------------------- #


def test_load_config_yaml(tmp_path: Path) -> None:
    p = tmp_path / "train.yaml"
    p.write_text("lr: 0.001\nbatch_size: 64\n", encoding="utf-8")
    assert load_config(p) == {"lr": 0.001, "batch_size": 64}


def test_load_config_json(tmp_path: Path) -> None:
    p = tmp_path / "train.json"
    p.write_text('{"lr": 0.001, "batch_size": 64}', encoding="utf-8")
    assert load_config(p) == {"lr": 0.001, "batch_size": 64}


def test_load_config_empty_yaml_returns_empty_dict(tmp_path: Path) -> None:
    p = tmp_path / "train.yaml"
    p.write_text("", encoding="utf-8")
    assert load_config(p) == {}


def test_load_config_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigEditError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_load_config_non_mapping_raises(tmp_path: Path) -> None:
    p = tmp_path / "train.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigEditError, match="mapping"):
        load_config(p)


def test_write_config_roundtrips_yaml(tmp_path: Path) -> None:
    p = tmp_path / "train.yaml"
    write_config(p, {"lr": 0.001, "dropout": 0.3})
    assert load_config(p) == {"lr": 0.001, "dropout": 0.3}


def test_write_config_roundtrips_json(tmp_path: Path) -> None:
    p = tmp_path / "train.json"
    write_config(p, {"lr": 0.001, "dropout": 0.3})
    assert load_config(p) == {"lr": 0.001, "dropout": 0.3}


# --------------------------------------------------------------------------- #
# apply_edits — happy path                                                    #
# --------------------------------------------------------------------------- #


def test_apply_edits_applies_confirmed_change(tmp_path: Path) -> None:
    p = tmp_path / "train.yaml"
    p.write_text("dropout: 0.1\nlr: 0.001\n", encoding="utf-8")

    edits = [
        ConfigEdit(key="dropout", current_value=0.1, proposed_value=0.3, rationale="overfit"),
    ]
    result = apply_edits(p, edits, confirm_fn=_yes)

    assert len(result.applied) == 1
    assert result.applied[0].key == "dropout"
    assert load_config(p)["dropout"] == 0.3
    # Backup was written and still has the old value.
    assert result.backup_path is not None
    assert load_config(result.backup_path)["dropout"] == 0.1


def test_apply_edits_rejection_keeps_file_unchanged(tmp_path: Path) -> None:
    p = tmp_path / "train.yaml"
    p.write_text("dropout: 0.1\n", encoding="utf-8")

    edits = [ConfigEdit(key="dropout", current_value=0.1, proposed_value=0.3)]
    result = apply_edits(p, edits, confirm_fn=_no)

    assert result.applied == []
    assert len(result.rejected) == 1
    assert load_config(p)["dropout"] == 0.1
    # No backup written when nothing was applied.
    assert result.backup_path is None


def test_apply_edits_accepts_dict_inputs(tmp_path: Path) -> None:
    p = tmp_path / "train.yaml"
    p.write_text("dropout: 0.1\n", encoding="utf-8")

    edits = [{"key": "dropout", "current_value": 0.1, "proposed_value": 0.3, "rationale": "x"}]
    result = apply_edits(p, edits, confirm_fn=_yes)
    assert len(result.applied) == 1


# --------------------------------------------------------------------------- #
# apply_edits — skip semantics                                                #
# --------------------------------------------------------------------------- #


def test_apply_edits_skips_when_value_already_matches(tmp_path: Path) -> None:
    p = tmp_path / "train.yaml"
    p.write_text("dropout: 0.3\n", encoding="utf-8")

    edits = [ConfigEdit(key="dropout", current_value=0.1, proposed_value=0.3)]
    result = apply_edits(p, edits, confirm_fn=_yes)

    assert result.applied == []
    assert len(result.skipped) == 1
    assert result.backup_path is None


def test_apply_edits_skips_when_key_missing(tmp_path: Path) -> None:
    p = tmp_path / "train.yaml"
    p.write_text("lr: 0.001\n", encoding="utf-8")

    edits = [ConfigEdit(key="dropout", current_value=0.1, proposed_value=0.3)]
    result = apply_edits(p, edits, confirm_fn=_yes)

    assert result.applied == []
    assert len(result.skipped) == 1


# --------------------------------------------------------------------------- #
# Multiple edits + dotted paths + backup                                      #
# --------------------------------------------------------------------------- #


def test_apply_edits_handles_multiple_edits(tmp_path: Path) -> None:
    p = tmp_path / "train.yaml"
    p.write_text("dropout: 0.1\nlr: 0.001\n", encoding="utf-8")

    edits = [
        ConfigEdit(key="dropout", current_value=0.1, proposed_value=0.3),
        ConfigEdit(key="lr", current_value=0.001, proposed_value=0.0005),
    ]
    result = apply_edits(p, edits, confirm_fn=_yes)
    data = load_config(p)
    assert data == {"dropout": 0.3, "lr": 0.0005}
    assert len(result.applied) == 2
    # Only one backup file regardless of how many edits applied.
    assert result.backup_path is not None


def test_apply_edits_writes_backup_only_once_per_session(tmp_path: Path) -> None:
    p = tmp_path / "train.yaml"
    p.write_text("dropout: 0.1\nlr: 0.001\n", encoding="utf-8")

    edits = [
        ConfigEdit(key="dropout", current_value=0.1, proposed_value=0.3),
        ConfigEdit(key="lr", current_value=0.001, proposed_value=0.0005),
    ]
    result = apply_edits(p, edits, confirm_fn=_yes)
    backups = list(tmp_path.glob("*.bak"))
    assert len(backups) == 1
    assert result.backup_path == backups[0]


def test_apply_edits_uses_dotted_path_for_nested_keys(tmp_path: Path) -> None:
    p = tmp_path / "train.yaml"
    p.write_text(
        yaml.safe_dump({"optim": {"lr": 0.001, "wd": 0.0}}),
        encoding="utf-8",
    )

    edits = [
        ConfigEdit(key="optim.lr", current_value=0.001, proposed_value=0.0005),
    ]
    result = apply_edits(p, edits, confirm_fn=_yes)
    assert len(result.applied) == 1
    assert load_config(p)["optim"]["lr"] == 0.0005


def test_apply_edits_surfaces_live_value_to_confirm_fn(tmp_path: Path) -> None:
    """If the live config diverges from the agent's reported current_value,
    the confirm_fn should see the live one (so the user can spot drift)."""
    p = tmp_path / "train.yaml"
    p.write_text("dropout: 0.2\n", encoding="utf-8")

    seen_current: list = []

    def spy(edit: ConfigEdit, _config: dict) -> bool:
        seen_current.append(edit.current_value)
        return True

    edits = [ConfigEdit(key="dropout", current_value=0.1, proposed_value=0.3)]
    apply_edits(p, edits, confirm_fn=spy)
    assert seen_current == [0.2]
