"""The one-difference guarantee of `ab_protocol.md` section 2.

Both arms get the same dataset, task, model and settings; they differ in
exactly one thing, the mlcompass block. That claim is the whole experiment —
if a stray token diverges, the measured difference is no longer attributable
to the intervention and every row in `ab_results.csv` becomes uninterpretable.

So it is asserted mechanically rather than trusted: for the same cell, the
control prompt must be a *strict* substring of the treatment prompt. When it
is not, the failure says where the two diverge instead of printing two walls
of text and leaving the reader to diff them by eye.

The process-defect checklist gets the same treatment. Section 4 calls it
frozen and deterministic, so it is tested against scripts whose defects are
known by construction.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmark" / "run_ab.py"

_spec = importlib.util.spec_from_file_location("run_ab", SCRIPT)
assert _spec is not None and _spec.loader is not None
run_ab = importlib.util.module_from_spec(_spec)
sys.modules["run_ab"] = run_ab
_spec.loader.exec_module(run_ab)


CELL = {
    "csv_path": r"C:\tmp\mlcab_abc123\train.csv",
    "target": "Class",
    "columns": "  - V1 (int64)\n  - V2 (int64)\n  - Class (int64)",
}

ADVISE = """\
⚠ Warnings
  • 215 exact duplicate row(s) (28.7% of the data). A random split puts copies
of the same row in both train and test.
"""


def _divergence(control: str, treatment: str) -> str:
    """Explain, in one paragraph, where the two prompts stop agreeing."""
    limit = min(len(control), len(treatment))
    index = next((i for i in range(limit) if control[i] != treatment[i]), limit)
    if index == limit and len(treatment) >= len(control):
        return (
            "the prompts agree on every shared character, so the divergence is "
            "not a prefix mismatch. Control is not a substring for another "
            f"reason (control {len(control)} chars, treatment {len(treatment)})."
        )
    line = control.count("\n", 0, index) + 1
    return (
        f"the prompts diverge at character {index} (line {line}). "
        f"control has {control[index : index + 60]!r}; "
        f"treatment has {treatment[index : index + 60]!r}. "
        "ab_protocol.md section 2 allows exactly one difference between the "
        "arms, the appended mlcompass block, so the treatment prompt must "
        "begin with the control prompt verbatim."
    )


@pytest.mark.parametrize(
    ("advise", "audit", "label"),
    [
        (ADVISE, "", "findings present"),
        ("", "", "no findings — section 2's empty block"),
        ("  • 13 exact duplicate row(s) (2.2% of the data).\n", "", "small findings"),
    ],
)
def test_control_prompt_is_a_strict_substring_of_treatment(
    advise: str, audit: str, label: str
) -> None:
    control, treatment = run_ab.build_prompts(
        csv_path=CELL["csv_path"],
        target=CELL["target"],
        columns=CELL["columns"],
        advise_stdout=advise,
        audit_stdout=audit,
    )
    assert control in treatment, f"[{label}] {_divergence(control, treatment)}"
    assert len(treatment) > len(control), (
        f"[{label}] the treatment prompt is not strictly longer than the control "
        f"prompt ({len(treatment)} vs {len(control)} characters). The arms would "
        "then be identical and the experiment would measure nothing."
    )
    assert treatment.startswith(control), (
        f"[{label}] control is a substring but not a prefix of treatment, which "
        "means the block was interleaved rather than appended: "
        f"{_divergence(control, treatment)}"
    )


def test_the_only_difference_is_the_mlcompass_block() -> None:
    """Removing the block from the treatment prompt returns the control prompt."""
    control, treatment = run_ab.build_prompts(
        csv_path=CELL["csv_path"],
        target=CELL["target"],
        columns=CELL["columns"],
        advise_stdout=ADVISE,
        audit_stdout="",
    )
    remainder = treatment[len(control) :]
    assert ADVISE in remainder
    assert treatment.replace(remainder, "") == control
    # The block must not smuggle in advice of its own: everything in it that is
    # not mlcompass's bytes is structural framing.
    framing = remainder.replace(ADVISE, "")
    assert "duplicate" not in framing.lower()
    assert "seed" not in framing.lower()
    assert "leak" not in framing.lower()


def test_an_empty_findings_block_is_still_present() -> None:
    """Section 2: no findings means an empty block, never a hint."""
    control, treatment = run_ab.build_prompts(
        csv_path=CELL["csv_path"],
        target=CELL["target"],
        columns=CELL["columns"],
        advise_stdout="",
        audit_stdout="",
    )
    remainder = treatment[len(control) :]
    assert "mlcompass advise" in remainder
    assert "mlcompass audit" in remainder
    assert not any(
        word in remainder.lower() for word in ("should", "recommend", "make sure", "remember")
    )


def test_a_per_arm_workspace_path_would_break_the_guarantee() -> None:
    """The regression that caught this harness out on its first live run.

    The train.csv path is inside the prompt. When each arm got its own
    `mkdtemp` workspace, the two prompts differed in the directory name as
    well as in the mlcompass block — a second difference, invisible to a test
    that builds both prompts from one path. `run_cell` now takes a workspace
    shared by the arms of a cell and re-asserts the relation at runtime; this
    fixes the shape of the failure it is guarding against.
    """
    control, _ = run_ab.build_prompts(
        csv_path=r"C:\tmp\mlcab_aaaaaaa\train.csv",
        target=CELL["target"],
        columns=CELL["columns"],
        advise_stdout="",
        audit_stdout="",
    )
    _, treatment = run_ab.build_prompts(
        csv_path=r"C:\tmp\mlcab_bbbbbbb\train.csv",
        target=CELL["target"],
        columns=CELL["columns"],
        advise_stdout=ADVISE,
        audit_stdout="",
    )
    assert not treatment.startswith(control)
    assert "diverge at character" in _divergence(control, treatment)


# --------------------------------------------------------------------------- #
# The frozen checklist                                                        #
# --------------------------------------------------------------------------- #

CLEAN = """\
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

np.random.seed(0)
df = pd.read_csv("train.csv")
df = df.drop_duplicates()
X = df.drop(columns=["Class"])
y = df["Class"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_tr, y_tr)
print(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
"""

DIRTY = """\
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("train.csv")
scaler = StandardScaler()
Xs = scaler.fit_transform(df)
model = LogisticRegression()
model.fit(Xs, df["Class"])
print(accuracy_score(df["Class"], model.predict(Xs)))
"""


def test_clean_script_scores_zero_defects() -> None:
    result = run_ab.check_defects(CLEAN, target="Class", duplicate_rows=215, minority_fraction=0.24)
    assert result["flags"] == {d: False for d in run_ab.DEFECT_IDS}, result["flags"]
    assert result["defect_count"] == 0


def test_dirty_script_scores_every_defect_its_data_allows() -> None:
    result = run_ab.check_defects(DIRTY, target="Class", duplicate_rows=215, minority_fraction=0.15)
    assert result["flags"] == {
        "no_seed": True,
        "leak_fit_before_split": True,
        "leak_duplicate_rows": True,
        "wrong_metric_for_imbalance": True,
        "no_validation": True,
        "target_in_features": True,
    }, result["flags"]
    assert result["defect_count"] == 6


def test_data_dependent_defects_cannot_fire_without_the_data_condition() -> None:
    """Two of the six are defined against the input, not the source alone."""
    result = run_ab.check_defects(DIRTY, target="Class", duplicate_rows=0, minority_fraction=0.48)
    assert result["flags"]["leak_duplicate_rows"] is False
    assert result["flags"]["wrong_metric_for_imbalance"] is False
    assert result["defect_count"] == 4


# The two scripts below are verbatim from the first live A/B pair on
# openml-1464 with qwen2.5:7b. Each exposed a real bug in the checker, so each
# is pinned here as the regression rather than paraphrased.

LIVE_CONTROL = """\
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")

X = df[['V1', 'V2', 'V3', 'V4']]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

import joblib
joblib.dump(model, 'trained_model.pkl')
"""

LIVE_TREATMENT = """\
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

data = pd.read_csv("train.csv")

X = data.drop(columns=['Class'])
y = data['Class']

# Handle exact duplicate rows (as suggested by the advisor)
data = data.drop_duplicates()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
dump(model, 'trained_model.joblib')
"""


def test_an_explicit_column_list_excludes_the_target() -> None:
    """`X = df[['V1', 'V2', 'V3', 'V4']]` does not leave Class in the features.

    The first version of the checker only recognised `.drop`, and reported a
    defect against a script that had done nothing wrong.
    """
    result = run_ab.check_defects(
        LIVE_CONTROL, target="Class", duplicate_rows=142, minority_fraction=0.239
    )
    assert result["flags"]["target_in_features"] is False
    assert result["flags"]["leak_duplicate_rows"] is True  # it never de-duplicates


def test_a_dedupe_after_the_features_are_taken_does_not_count() -> None:
    """De-duplicating a frame the split no longer reads removes nothing.

    This script calls `drop_duplicates()` before the `train_test_split` line
    but after `X` and `y` were taken from the frame, so nothing that is split
    has had a duplicate removed. Checking line order against the split call
    alone credited it; the deadline is the derivation of the split's inputs.
    """
    result = run_ab.check_defects(
        LIVE_TREATMENT, target="Class", duplicate_rows=142, minority_fraction=0.239
    )
    assert result["flags"]["leak_duplicate_rows"] is True
    assert result["dedupe_deadline"] == 8  # the line assigning X


def test_a_dedupe_before_the_features_are_taken_does_count() -> None:
    late_dedupe = (
        "\n# Handle exact duplicate rows (as suggested by the advisor)"
        "\ndata = data.drop_duplicates()\n"
    )
    fixed = LIVE_TREATMENT.replace(
        "X = data.drop(columns=['Class'])",
        "data = data.drop_duplicates()\nX = data.drop(columns=['Class'])",
    ).replace(late_dedupe, "")
    result = run_ab.check_defects(
        fixed, target="Class", duplicate_rows=142, minority_fraction=0.239
    )
    assert result["flags"]["leak_duplicate_rows"] is False


def test_a_positional_slice_excludes_the_target_only_when_it_is_last() -> None:
    script = CLEAN.replace('X = df.drop(columns=["Class"])', "X = df.iloc[:, :-1]")
    last = run_ab.check_defects(
        script, target="Class", duplicate_rows=0, minority_fraction=0.4, target_is_last_column=True
    )
    not_last = run_ab.check_defects(
        script, target="Class", duplicate_rows=0, minority_fraction=0.4, target_is_last_column=False
    )
    assert last["flags"]["target_in_features"] is False
    assert not_last["flags"]["target_in_features"] is True


def test_a_column_list_containing_the_target_is_not_an_exclusion() -> None:
    script = LIVE_CONTROL.replace("[['V1', 'V2', 'V3', 'V4']]", "[['V1', 'V2', 'Class']]")
    result = run_ab.check_defects(script, target="Class", duplicate_rows=0, minority_fraction=0.4)
    assert result["flags"]["target_in_features"] is True


def test_the_checklist_is_deterministic() -> None:
    first = run_ab.check_defects(DIRTY, target="Class", duplicate_rows=9, minority_fraction=0.1)
    second = run_ab.check_defects(DIRTY, target="Class", duplicate_rows=9, minority_fraction=0.1)
    assert first == second


@pytest.mark.parametrize(
    ("reply", "expected_head"),
    [
        ("Here you go:\n```python\nimport pandas as pd\n```\nEnjoy.", "import pandas"),
        ("```\nimport pandas as pd\n```", "import pandas"),
        ("import pandas as pd\nprint(1)\n", "import pandas"),
    ],
)
def test_extract_python_finds_the_script(reply: str, expected_head: str) -> None:
    source, _how = run_ab.extract_python(reply)
    assert source is not None
    assert source.startswith(expected_head)


def test_extract_python_reports_a_reply_with_no_code() -> None:
    source, how = run_ab.extract_python("I am sorry, I cannot help with that.")
    assert source is None
    assert "no Python" in how


def test_extract_python_keeps_a_truncated_block() -> None:
    """A cut-off reply is kept and allowed to fail at execution, not discarded."""
    source, how = run_ab.extract_python("```python\nimport pandas as pd\ndf = pd.read")
    assert source is not None
    assert "unclosed fence" in how
