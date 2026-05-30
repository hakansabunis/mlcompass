"""Tests for the static script analyzer (``tools.script.audit_script``)."""

from __future__ import annotations

from pathlib import Path

from mlcompass.tools.script import ALL_RULE_IDS, audit_script

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _write(tmp_path: Path, code: str, name: str = "train.py") -> Path:
    """Write a snippet to a temp file and return its path."""
    p = tmp_path / name
    p.write_text(code, encoding="utf-8")
    return p


def _rule_ids(result: dict) -> list[str]:
    """Convenience: list of rule_ids in the findings, in order."""
    return [f["rule_id"] for f in result["findings"]]


# --------------------------------------------------------------------------- #
# Plumbing                                                                    #
# --------------------------------------------------------------------------- #


def test_audit_returns_top_level_keys(tmp_path: Path) -> None:
    p = _write(tmp_path, "x = 1\n")
    result = audit_script(p)
    for key in ("path", "lines", "frameworks", "findings"):
        assert key in result


def test_empty_script_produces_no_findings(tmp_path: Path) -> None:
    p = _write(tmp_path, "")
    result = audit_script(p)
    assert result["findings"] == []
    assert result["lines"] == 0


def test_no_ml_frameworks_skips_seed_check(tmp_path: Path) -> None:
    p = _write(tmp_path, "print('hello')\n")
    result = audit_script(p)
    assert "seed" not in _rule_ids(result)


def test_frameworks_are_detected(tmp_path: Path) -> None:
    code = (
        "import torch\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\n"
    )
    result = audit_script(_write(tmp_path, code))
    assert "torch" in result["frameworks"]
    assert "numpy" in result["frameworks"]
    assert "sklearn" in result["frameworks"]


def test_findings_are_sorted_by_severity_then_line(tmp_path: Path) -> None:
    # info first, error second, warning third in source — must come out err, warn, info
    code = """
import torch

# This will trigger eval_mode warning (info-ish? actually warning)
model = torch.nn.Linear(10, 1)
model.train()

# Adam with momentum — error
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, momentum=0.9)

# Tiny batch — info
loader = torch.utils.data.DataLoader(None, batch_size=1)
"""
    result = audit_script(_write(tmp_path, code))
    severities = [f["severity"] for f in result["findings"]]
    # The order should respect: error < warning < info.
    assert severities == sorted(
        severities,
        key=lambda s: {"error": 0, "warning": 1, "info": 2}.get(s, 99),
    )


# --------------------------------------------------------------------------- #
# Rule: seed                                                                  #
# --------------------------------------------------------------------------- #


def test_missing_seed_triggers_error(tmp_path: Path) -> None:
    code = """
import torch
import torch.nn as nn
model = nn.Linear(10, 1)
"""
    result = audit_script(_write(tmp_path, code))
    seed_findings = [f for f in result["findings"] if f["rule_id"] == "seed"]
    assert len(seed_findings) == 1
    assert seed_findings[0]["severity"] == "error"


def test_torch_manual_seed_satisfies_rule(tmp_path: Path) -> None:
    code = """
import torch
torch.manual_seed(42)
"""
    result = audit_script(_write(tmp_path, code))
    assert "seed" not in _rule_ids(result)


def test_numpy_random_seed_also_satisfies_rule(tmp_path: Path) -> None:
    code = """
import numpy as np
np.random.seed(42)
"""
    result = audit_script(_write(tmp_path, code))
    assert "seed" not in _rule_ids(result)


def test_pl_seed_everything_satisfies_rule(tmp_path: Path) -> None:
    code = """
import torch
import pytorch_lightning as pl
pl.seed_everything(42)
"""
    result = audit_script(_write(tmp_path, code))
    assert "seed" not in _rule_ids(result)


# --------------------------------------------------------------------------- #
# Rule: val_split                                                             #
# --------------------------------------------------------------------------- #


def test_missing_val_split_warns(tmp_path: Path) -> None:
    code = "import torch\ntorch.manual_seed(0)\nmodel = torch.nn.Linear(10, 1)\n"
    result = audit_script(_write(tmp_path, code))
    val_findings = [f for f in result["findings"] if f["rule_id"] == "val_split"]
    assert any(f["severity"] == "warning" for f in val_findings)


def test_train_test_split_satisfies_rule(tmp_path: Path) -> None:
    code = """
import torch
torch.manual_seed(0)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
"""
    result = audit_script(_write(tmp_path, code))
    # No "missing val split" warning; size is fine.
    assert "val_split" not in _rule_ids(result)


def test_tiny_val_split_warns(tmp_path: Path) -> None:
    code = """
import torch
torch.manual_seed(0)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01)
"""
    result = audit_script(_write(tmp_path, code))
    val_findings = [f for f in result["findings"] if f["rule_id"] == "val_split"]
    assert any("small" in f["message"].lower() for f in val_findings)


# --------------------------------------------------------------------------- #
# Rule: optimizer                                                             #
# --------------------------------------------------------------------------- #


def test_adam_with_momentum_is_error(tmp_path: Path) -> None:
    code = """
import torch
torch.manual_seed(0)
optimizer = torch.optim.Adam(params, lr=1e-3, momentum=0.9)
"""
    result = audit_script(_write(tmp_path, code))
    opt_findings = [f for f in result["findings"] if f["rule_id"] == "optimizer"]
    assert any(f["severity"] == "error" and "momentum" in f["message"] for f in opt_findings)


def test_sgd_without_momentum_is_info(tmp_path: Path) -> None:
    code = """
import torch
torch.manual_seed(0)
optimizer = torch.optim.SGD(params, lr=0.01)
"""
    result = audit_script(_write(tmp_path, code))
    opt_findings = [f for f in result["findings"] if f["rule_id"] == "optimizer"]
    assert any(f["severity"] == "info" for f in opt_findings)


def test_high_lr_warns(tmp_path: Path) -> None:
    code = """
import torch
torch.manual_seed(0)
optimizer = torch.optim.AdamW(params, lr=5.0)
"""
    result = audit_script(_write(tmp_path, code))
    opt_findings = [f for f in result["findings"] if f["rule_id"] == "optimizer"]
    assert any("high" in f["message"].lower() for f in opt_findings)


def test_low_lr_warns(tmp_path: Path) -> None:
    code = """
import torch
torch.manual_seed(0)
optimizer = torch.optim.AdamW(params, lr=1e-9)
"""
    result = audit_script(_write(tmp_path, code))
    opt_findings = [f for f in result["findings"] if f["rule_id"] == "optimizer"]
    assert any("small" in f["message"].lower() for f in opt_findings)


# --------------------------------------------------------------------------- #
# Rule: loss_stability                                                        #
# --------------------------------------------------------------------------- #


def test_torch_log_without_clamp_warns(tmp_path: Path) -> None:
    code = """
import torch
torch.manual_seed(0)
y = torch.log(x)
"""
    result = audit_script(_write(tmp_path, code))
    assert any(f["rule_id"] == "loss_stability" for f in result["findings"])


def test_torch_log_with_clamp_is_silent(tmp_path: Path) -> None:
    code = """
import torch
torch.manual_seed(0)
y = torch.log(x.clamp(min=1e-7))
"""
    result = audit_script(_write(tmp_path, code))
    assert "loss_stability" not in _rule_ids(result)


def test_torch_log_with_added_epsilon_is_silent(tmp_path: Path) -> None:
    code = """
import torch
torch.manual_seed(0)
y = torch.log(x + 1e-7)
"""
    result = audit_script(_write(tmp_path, code))
    assert "loss_stability" not in _rule_ids(result)


# --------------------------------------------------------------------------- #
# Rule: dataloader                                                            #
# --------------------------------------------------------------------------- #


def test_dataloader_without_shuffle_warns(tmp_path: Path) -> None:
    code = """
import torch
torch.manual_seed(0)
loader = torch.utils.data.DataLoader(dataset, batch_size=32)
"""
    result = audit_script(_write(tmp_path, code))
    assert any(f["rule_id"] == "dataloader" for f in result["findings"])


def test_dataloader_with_shuffle_kw_is_silent(tmp_path: Path) -> None:
    code = """
import torch
torch.manual_seed(0)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
"""
    result = audit_script(_write(tmp_path, code))
    assert "dataloader" not in _rule_ids(result)


# --------------------------------------------------------------------------- #
# Rule: grad_clipping                                                         #
# --------------------------------------------------------------------------- #


def test_rnn_without_grad_clip_warns(tmp_path: Path) -> None:
    code = """
import torch
import torch.nn as nn
torch.manual_seed(0)
model = nn.LSTM(10, 20)
"""
    result = audit_script(_write(tmp_path, code))
    assert any(f["rule_id"] == "grad_clipping" for f in result["findings"])


def test_rnn_with_clip_grad_norm_is_silent(tmp_path: Path) -> None:
    code = """
import torch
import torch.nn as nn
torch.manual_seed(0)
model = nn.LSTM(10, 20)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
"""
    result = audit_script(_write(tmp_path, code))
    assert "grad_clipping" not in _rule_ids(result)


def test_no_rnn_no_grad_clip_warning(tmp_path: Path) -> None:
    code = """
import torch
import torch.nn as nn
torch.manual_seed(0)
model = nn.Linear(10, 1)
"""
    result = audit_script(_write(tmp_path, code))
    assert "grad_clipping" not in _rule_ids(result)


# --------------------------------------------------------------------------- #
# Rule: eval_mode                                                             #
# --------------------------------------------------------------------------- #


def test_train_without_eval_warns(tmp_path: Path) -> None:
    code = """
import torch
torch.manual_seed(0)
model = torch.nn.Linear(10, 1)
model.train()
"""
    result = audit_script(_write(tmp_path, code))
    assert any(f["rule_id"] == "eval_mode" for f in result["findings"])


def test_train_with_eval_is_silent(tmp_path: Path) -> None:
    code = """
import torch
torch.manual_seed(0)
model = torch.nn.Linear(10, 1)
model.train()
model.eval()
"""
    result = audit_script(_write(tmp_path, code))
    assert "eval_mode" not in _rule_ids(result)


# --------------------------------------------------------------------------- #
# Rule: batch_size                                                            #
# --------------------------------------------------------------------------- #


def test_tiny_batch_size_is_info(tmp_path: Path) -> None:
    code = """
import torch
torch.manual_seed(0)
loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
"""
    result = audit_script(_write(tmp_path, code))
    bs_findings = [f for f in result["findings"] if f["rule_id"] == "batch_size"]
    assert any(f["severity"] == "info" for f in bs_findings)


def test_huge_batch_size_is_info(tmp_path: Path) -> None:
    code = """
import torch
torch.manual_seed(0)
loader = torch.utils.data.DataLoader(ds, batch_size=8192, shuffle=True)
"""
    result = audit_script(_write(tmp_path, code))
    bs_findings = [f for f in result["findings"] if f["rule_id"] == "batch_size"]
    assert any(f["severity"] == "info" for f in bs_findings)


def test_reasonable_batch_size_is_silent(tmp_path: Path) -> None:
    code = """
import torch
torch.manual_seed(0)
loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
"""
    result = audit_script(_write(tmp_path, code))
    assert "batch_size" not in _rule_ids(result)


# --------------------------------------------------------------------------- #
# Skip rules                                                                  #
# --------------------------------------------------------------------------- #


def test_skip_rules_filters_findings(tmp_path: Path) -> None:
    code = "import torch\nmodel = torch.nn.Linear(10, 1)\n"
    full = audit_script(_write(tmp_path, code))
    filtered = audit_script(_write(tmp_path, code), skip_rules={"seed"})
    assert "seed" in _rule_ids(full)
    assert "seed" not in _rule_ids(filtered)


def test_all_rule_ids_are_skippable(tmp_path: Path) -> None:
    code = """
import torch
import torch.nn as nn
model = nn.LSTM(10, 20)
model.train()
optimizer = torch.optim.Adam(params, lr=1e-3, momentum=0.9)
loader = torch.utils.data.DataLoader(ds, batch_size=1)
y = torch.log(x)
"""
    result = audit_script(_write(tmp_path, code), skip_rules=set(ALL_RULE_IDS))
    assert result["findings"] == []
