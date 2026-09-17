"""The `target_in_features` rule, pinned form by form.

This rule scores the *absence* of every enumerated way of taking the target
out of the feature matrix. That shape is the reason it needs this file: the
list of forms is open-ended, so every form nobody enumerated becomes a false
positive on correct code, and a false positive here inflates the defect count
the A/B benchmark reports.

It has already happened once. Yusuf Ünlü found, in review of the 108-run
battery, that a script excluding the target with a list comprehension

    feature_cols = [c for c in df.columns if c != target_col]

was scored `target_in_features: yes`. Fifteen of the 108 preserved runs
carried that false positive (amendment A7).

So: every accepted form gets a test, and every genuine defect gets one too,
because widening an exclusion rule is exactly how you turn a detector into a
rubber stamp. Add the test before the form.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

from run_ab import check_defects  # noqa: E402

TARGET = "Class"

HEAD = """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("train.csv")
"""


def _flag(body: str) -> bool:
    """Run the checker over HEAD + body and return the target_in_features flag."""
    report = check_defects(
        HEAD + body,
        target=TARGET,
        duplicate_rows=0,
        minority_fraction=None,
        target_is_last_column=False,
    )
    return bool(report["flags"]["target_in_features"])


# --------------------------------------------------------------------------- #
# Accepted: the target really is excluded. The flag must be False.            #
# --------------------------------------------------------------------------- #

EXCLUDES = {
    "drop_literal": 'X = df.drop(columns=["Class"])\ny = df["Class"]\n',
    "drop_via_variable": (
        'target_col = "Class"\nX = df.drop(columns=[target_col])\ny = df[target_col]\n'
    ),
    "pop": 'y = df.pop("Class")\nX = df\n',
    "columns_neq": 'X = df.loc[:, df.columns != "Class"]\ny = df["Class"]\n',
    "columns_drop": 'X = df[df.columns.drop("Class")]\ny = df["Class"]\n',
    "columns_difference": 'X = df[df.columns.difference(["Class"])]\ny = df["Class"]\n',
    "explicit_list": 'X = df[["V1", "V2", "V3"]]\ny = df["Class"]\n',
    # Form 6 - the one that was missing.
    "comprehension_neq_variable": (
        'target_col = "Class"\n'
        "feature_cols = [c for c in df.columns if c != target_col]\n"
        "X = df[feature_cols]\ny = df[target_col]\n"
    ),
    "comprehension_neq_literal": (
        'feature_cols = [c for c in df.columns if c != "Class"]\n'
        'X = df[feature_cols]\ny = df["Class"]\n'
    ),
    "comprehension_not_in": (
        'target_col = "Class"\n'
        "X = df[[c for c in df.columns if c not in [target_col]]]\n"
        "y = df[target_col]\n"
    ),
    "comprehension_inline": ('X = df[[c for c in df.columns if c != "Class"]]\ny = df["Class"]\n'),
    # Form 7.
    "set_difference": (
        'feature_cols = list(set(df.columns) - {"Class"})\nX = df[feature_cols]\ny = df["Class"]\n'
    ),
}


@pytest.mark.parametrize("name", sorted(EXCLUDES))
def test_excluded_target_is_not_flagged(name: str) -> None:
    assert not _flag(EXCLUDES[name]), (
        f"{name}: the script excludes the target, so target_in_features must be "
        "False. A true flag here is a false positive that inflates the A/B "
        "defect count."
    )


# --------------------------------------------------------------------------- #
# Rejected: the target really is in the features. The flag must be True.      #
# Widening the rule must not cost us these.                                   #
# --------------------------------------------------------------------------- #

LEAVES_TARGET_IN = {
    "whole_frame": 'X = df\ny = df["Class"]\n',
    "comprehension_without_filter": (
        'feature_cols = [c for c in df.columns]\nX = df[feature_cols]\ny = df["Class"]\n'
    ),
    # Filters a different column - the target stays in.
    "comprehension_filters_other_column": (
        'feature_cols = [c for c in df.columns if c != "V1"]\n'
        'X = df[feature_cols]\ny = df["Class"]\n'
    ),
    # A comparison against the target that is not a column filter.
    "unrelated_comparison": (
        'target_col = "Class"\n'
        'rows = [r for r in df.index if df.loc[r, "V1"] != 0]\n'
        "X = df\ny = df[target_col]\n"
    ),
    "explicit_list_including_target": 'X = df[["V1", "V2", "Class"]]\ny = df["Class"]\n',
}


@pytest.mark.parametrize("name", sorted(LEAVES_TARGET_IN))
def test_target_left_in_is_flagged(name: str) -> None:
    assert _flag(LEAVES_TARGET_IN[name]), (
        f"{name}: the target is still in the feature matrix, so "
        "target_in_features must be True. A false flag here means the rule was "
        "widened into a rubber stamp."
    )


def test_the_run_that_started_this() -> None:
    """The exact script from the review, byte for byte.

    runs/ab-20260915-3b0221-1464-control-deepseek-flash-r1/emitted.py:11-15.
    Kept as its own test because a parametrised case can drift away from what
    the model actually wrote; this one cannot.
    """
    source = HEAD + (
        'target_col = "Class"\n'
        "feature_cols = [c for c in df.columns if c != target_col]\n"
        "\n"
        "X = df[feature_cols]\n"
        "y = df[target_col]\n"
    )
    report = check_defects(
        source,
        target=TARGET,
        duplicate_rows=0,
        minority_fraction=None,
        target_is_last_column=True,
    )
    assert not report["flags"]["target_in_features"]


# --------------------------------------------------------------------------- #
# Form 8 and the seed rule: constants the checker used to look straight past.  #
#                                                                             #
# Found running the A8 arm. Three of its 36 runs scored WORSE after the model  #
# revised its own script, which read as a real finding until the scripts were  #
# opened: two of the three were the checker missing a named constant, and only #
# one was a genuine regression. The bias has a direction that matters — a      #
# revised script is tidier, tidier code hoists lists and seeds to module       #
# constants, so the rule penalised exactly the arm under study.                #
# --------------------------------------------------------------------------- #

NAMED_CONSTANT_EXCLUDES = {
    # runs/ab-20260915-19fa60-1480-control+revise-deepseek-flash-r2/emitted.py
    "concatenated_constants": (
        'TARGET_COL = "Class"\n'
        'NUMERIC_COLS = ["V1", "V3", "V4"]\n'
        'CATEGORICAL_COLS = ["V2"]\n'
        "FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS\n"
        "X = df[FEATURE_COLS].copy()\n"
        "y = df[TARGET_COL].copy()\n"
    ),
    "single_constant": (
        'FEATURE_COLS = ["V1", "V2", "V3"]\nX = df[FEATURE_COLS]\ny = df["Class"]\n'
    ),
}


@pytest.mark.parametrize("name", sorted(NAMED_CONSTANT_EXCLUDES))
def test_a_named_feature_list_counts_as_excluding_the_target(name: str) -> None:
    assert not _flag(NAMED_CONSTANT_EXCLUDES[name]), (
        f"{name}: the named list omits the target, so the target is excluded. "
        "Form 4 already accepts this list written inline; hoisting it to a "
        "constant is the same code, tidier."
    )


def test_a_named_list_that_contains_the_target_is_still_flagged() -> None:
    body = 'FEATURE_COLS = ["V1", "V2", "Class"]\nX = df[FEATURE_COLS]\ny = df["Class"]\n'
    assert _flag(body)


def test_a_named_list_nobody_selects_with_proves_nothing() -> None:
    """A constant that is never used to build X says nothing about X."""
    body = 'FEATURE_COLS = ["V1", "V2"]\nX = df\ny = df["Class"]\n'
    assert _flag(body)


SEED_VIA_CONSTANT = """import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
RANDOM_STATE = 42
df = pd.read_csv("train.csv")
X = df.drop(columns=["Class"])
y = df["Class"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=RANDOM_STATE)
model = RandomForestClassifier(random_state=RANDOM_STATE).fit(X_tr, y_tr)
"""


def test_a_seed_hoisted_to_a_constant_counts_as_seeded() -> None:
    """runs/ab-20260915-19fa60-1480-control+revise-openai-gpt-5.4-mini-r3.

    The seed patterns want a digit. This script sets RANDOM_STATE = 42 and
    passes it in three places, and was scored unseeded.
    """
    report = check_defects(
        SEED_VIA_CONSTANT,
        target="Class",
        duplicate_rows=0,
        minority_fraction=None,
        target_is_last_column=True,
    )
    assert not report["flags"]["no_seed"]


def test_a_script_with_no_seed_anywhere_is_still_flagged() -> None:
    """Widening the seed rule must not make it unable to fire."""
    unseeded = (
        SEED_VIA_CONSTANT.replace("RANDOM_STATE = 42\n", "")
        .replace(", random_state=RANDOM_STATE", "")
        .replace("random_state=RANDOM_STATE", "")
    )
    report = check_defects(
        unseeded,
        target="Class",
        duplicate_rows=0,
        minority_fraction=None,
        target_is_last_column=True,
    )
    assert report["flags"]["no_seed"]


TARGET_VIA_PARAMETER_DEFAULT = """\
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def train_and_save_model(csv_path, target_column="price"):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
"""


def test_target_dropped_through_a_parameter_default_is_not_a_defect() -> None:
    """runs/ab-20260915-3b0221-44031-control-mistral-ministral-8b-r1 and -r2.

    The target name is bound by a function-parameter default, not by an
    assignment, so the line-anchored harvest never saw it and the script was
    scored as leaving the target in the features. It drops it on line 9.
    Found by the differential audit against the ast scorer (amendment A16).
    """
    report = check_defects(
        TARGET_VIA_PARAMETER_DEFAULT,
        target="price",
        duplicate_rows=0,
        minority_fraction=None,
        target_is_last_column=False,
    )
    assert not report["flags"]["target_in_features"]


def test_target_via_keyword_argument_is_not_a_defect() -> None:
    """The same binding at a call site rather than in a signature."""
    source = (
        TARGET_VIA_PARAMETER_DEFAULT.replace(
            'def train_and_save_model(csv_path, target_column="price"):',
            "def train_and_save_model(csv_path, target_column):",
        )
        + '\n\ntrain_and_save_model("t.csv", target_column="price")\n'
    )
    report = check_defects(
        source,
        target="price",
        duplicate_rows=0,
        minority_fraction=None,
        target_is_last_column=False,
    )
    assert not report["flags"]["target_in_features"]


def test_a_script_that_really_leaves_the_target_in_is_still_flagged() -> None:
    """Widening the harvest must not make the rule unable to fire."""
    leaky = TARGET_VIA_PARAMETER_DEFAULT.replace("X = df.drop(columns=[target_column])", "X = df")
    report = check_defects(
        leaky,
        target="price",
        duplicate_rows=0,
        minority_fraction=None,
        target_is_last_column=False,
    )
    assert report["flags"]["target_in_features"]
