"""Tests for the tabular ``pandas + sklearn`` rules in ``tools.script``.

Every rule here ships with a pair: a script that must trip it, and a
near-miss script that is close enough to be a plausible false positive
and must stay silent. The near-miss half is the load-bearing half --
``audit``'s known defect is false positives, not misses.
"""

from __future__ import annotations

from pathlib import Path

from mlcompass.tools.script import audit_script


def _write(tmp_path: Path, code: str, name: str = "train.py") -> Path:
    p = tmp_path / name
    p.write_text(code, encoding="utf-8")
    return p


def _rule_ids(result: dict) -> list[str]:
    return [f["rule_id"] for f in result["findings"]]


def _ids_for(result: dict, rule_id: str) -> list[dict]:
    return [f for f in result["findings"] if f["rule_id"] == rule_id]


# --------------------------------------------------------------------------- #
# Guard: a correct, idiomatic pandas + sklearn script must stay silent         #
# --------------------------------------------------------------------------- #


CLEAN_SKLEARN_SCRIPT = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
pipe.fit(X_train, y_train)

proba = pipe.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, proba))
print(classification_report(y_test, pipe.predict(X_test)))
"""


def test_correct_sklearn_script_has_no_findings(tmp_path: Path) -> None:
    """The whole point: a right script must draw nothing from any rule."""
    result = audit_script(_write(tmp_path, CLEAN_SKLEARN_SCRIPT))
    assert result["findings"] == [], result["findings"]


# --------------------------------------------------------------------------- #
# Rule: preprocess_leak                                                        #
# --------------------------------------------------------------------------- #


def test_scaler_fitted_before_split_is_error(tmp_path: Path) -> None:
    code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
"""
    found = _ids_for(audit_script(_write(tmp_path, code)), "preprocess_leak")
    assert len(found) == 1, found
    assert found[0]["severity"] == "error"
    assert found[0]["line"] == 12
    assert "StandardScaler" in found[0]["message"]


def test_scaler_fitted_after_split_is_silent(tmp_path: Path) -> None:
    """Near miss: same scaler, same names, correct order."""
    code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
"""
    assert "preprocess_leak" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_scaler_fit_inside_helper_defined_above_split_is_silent(tmp_path: Path) -> None:
    """Near miss: the fit is lexically above the split but executes after it."""
    code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def scale(frame):
    scaler = StandardScaler()
    return scaler.fit_transform(frame)


df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train = scale(X_train)
print(X_test, y_train, y_test)
"""
    assert "preprocess_leak" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_pipeline_built_before_split_is_silent(tmp_path: Path) -> None:
    """Near miss: StandardScaler() is constructed above the split but never fitted there."""
    code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
"""
    assert "preprocess_leak" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_imputer_fitted_before_cross_val_score_is_error(tmp_path: Path) -> None:
    code = """
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])

imp = SimpleImputer(strategy="mean")
X = imp.fit_transform(X)

print(cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5))
"""
    found = _ids_for(audit_script(_write(tmp_path, code)), "preprocess_leak")
    assert len(found) == 1, found
    assert found[0]["severity"] == "error"
    assert "SimpleImputer" in found[0]["message"]


def test_stateless_transformer_before_split_is_silent(tmp_path: Path) -> None:
    """Near miss: Normalizer is row-wise and FunctionTransformer is stateless."""
    code = """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, Normalizer

np.random.seed(0)
df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])

X = Normalizer().fit_transform(X)
X = FunctionTransformer(np.log1p).fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(X_train, X_test, y_train, y_test)
"""
    assert "preprocess_leak" not in _rule_ids(audit_script(_write(tmp_path, code)))


# --------------------------------------------------------------------------- #
# Rule: refit_across_split                                                     #
# --------------------------------------------------------------------------- #


def test_scaler_refitted_on_both_halves_is_error(tmp_path: Path) -> None:
    code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
print(y_train, y_test)
"""
    found = _ids_for(audit_script(_write(tmp_path, code)), "refit_across_split")
    assert len(found) == 1, found
    assert found[0]["severity"] == "error"
    assert found[0]["line"] == 13


def test_transform_on_test_half_is_silent(tmp_path: Path) -> None:
    """Near miss: four characters different, and correct."""
    code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(y_train, y_test)
"""
    assert "refit_across_split" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_calibration_split_workflow_is_silent(tmp_path: Path) -> None:
    """Near miss, found in the wild.

    Reduced from scikit-learn's own ``tests/test_calibration.py``, where an
    earlier version of this rule reported an error. Fitting a *calibrator* on
    the second split is the whole point of a train / calibrate / test
    workflow: a different object, deliberately fitted on held-out rows.
    """
    code = """
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
X_train, X_cal, y_train, y_cal = train_test_split(X, y, random_state=42)

clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

cal_clf = CalibratedClassifierCV(clf, cv=3)
cal_clf.fit(X_cal, y_cal)
print(cal_clf.predict_proba(X_cal))
"""
    assert "refit_across_split" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_same_name_in_another_function_is_silent(tmp_path: Path) -> None:
    """Near miss, found in the wild.

    Reduced from scikit-learn's ``feature_extraction/tests/test_text.py``,
    where an earlier version of this rule matched a plain local named
    ``test_data`` against a ``train_test_split`` 120 lines away in a
    different function. Split outputs are resolved per scope.
    """
    code = """
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def check_max_df():
    test_data = ["abc", "dea", "eat"]
    vect = CountVectorizer(analyzer="char", max_df=1.0)
    vect.fit(test_data)
    vect.max_df = 0.5
    vect.fit(test_data)
    return vect


def build():
    df = pd.read_csv("data.csv")
    target = df["label"]
    data = df.drop(columns=["label"])
    train_data, test_data, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=0
    )
    return train_data, test_data, y_train, y_test
"""
    assert "refit_across_split" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_fit_on_second_output_alone_is_silent(tmp_path: Path) -> None:
    """Recorded decision, not an oversight.

    Fitting on the second output and scoring on the first is reversed
    naming, not leakage -- the two sets are still disjoint -- so the rule
    stays out of it. ``test_size=0.8`` is the shape that makes this common.
    """
    code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
X_small, X_big, y_small, y_big = train_test_split(X, y, test_size=0.8, random_state=0)

model = LogisticRegression(max_iter=1000)
model.fit(X_big, y_big)
print(model.score(X_small, y_small))
"""
    assert "refit_across_split" not in _rule_ids(audit_script(_write(tmp_path, code)))


# --------------------------------------------------------------------------- #
# Rule: target_leak                                                            #
# --------------------------------------------------------------------------- #


def test_target_left_in_feature_frame_is_error(tmp_path: Path) -> None:
    code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")
X = df
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
"""
    found = _ids_for(audit_script(_write(tmp_path, code)), "target_leak")
    assert len(found) == 1, found
    assert found[0]["severity"] == "error"
    assert "target" in found[0]["message"]


def test_target_dropped_from_feature_frame_is_silent(tmp_path: Path) -> None:
    """Near miss: identical shape, with the one call that makes it correct."""
    code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
"""
    assert "target_leak" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_target_popped_from_feature_frame_is_silent(tmp_path: Path) -> None:
    """Near miss: ``X = df`` is correct here, because pop() already removed the column."""
    code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")
y = df.pop("target")
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
"""
    assert "target_leak" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_target_leak_via_direct_fit_is_error(tmp_path: Path) -> None:
    """No split at all -- the frame goes straight into fit()."""
    code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")
X = df.copy()
y = df["churn"]

model = RandomForestClassifier(random_state=0)
model.fit(X, y)
"""
    found = _ids_for(audit_script(_write(tmp_path, code)), "target_leak")
    assert len(found) == 1, found
    assert "churn" in found[0]["message"]


def test_column_subset_feature_frame_is_silent(tmp_path: Path) -> None:
    """Near miss: X is a column subset, so whether it holds the target is undecidable."""
    code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["age", "tenure", "charges"]
df = pd.read_csv("data.csv")
X = df[FEATURES]
y = df["churn"]

model = RandomForestClassifier(random_state=0)
model.fit(X, y)
"""
    assert "target_leak" not in _rule_ids(audit_script(_write(tmp_path, code)))


# --------------------------------------------------------------------------- #
# Rule: random_state                                                           #
# --------------------------------------------------------------------------- #


def test_train_test_split_without_random_state_warns(tmp_path: Path) -> None:
    code = """
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(len(X_train), len(X_test), len(y_train), len(y_test))
"""
    found = _ids_for(audit_script(_write(tmp_path, code)), "random_state")
    assert len(found) == 1, found
    assert found[0]["severity"] == "warning"
    assert found[0]["line"] == 8


def test_random_forest_without_random_state_warns(tmp_path: Path) -> None:
    code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
"""
    found = _ids_for(audit_script(_write(tmp_path, code)), "random_state")
    assert len(found) == 1, found
    assert "RandomForestClassifier" in found[0]["message"]


def test_random_state_supplied_is_silent(tmp_path: Path) -> None:
    code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = RandomForestClassifier(n_estimators=300, random_state=0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
"""
    assert "random_state" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_global_numpy_seed_suppresses_random_state(tmp_path: Path) -> None:
    """Near miss: sklearn draws from the global numpy RNG, so this script does reproduce."""
    code = """
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42)

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
"""
    assert "random_state" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_deterministic_estimators_are_not_flagged(tmp_path: Path) -> None:
    """Near miss: none of these are stochastic with their default parameters."""
    code = """
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
for model in (LinearRegression(), LogisticRegression(), Ridge(), KNeighborsClassifier(), SVC()):
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
"""
    assert "random_state" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_kfold_without_shuffle_is_not_flagged(tmp_path: Path) -> None:
    """Near miss: KFold(shuffle=False) is deterministic; sklearn rejects random_state there."""
    code = """
import pandas as pd
from sklearn.model_selection import KFold

df = pd.read_csv("data.csv")
cv = KFold(n_splits=5)
for train_idx, test_idx in cv.split(df):
    print(len(train_idx), len(test_idx))
"""
    assert "random_state" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_kfold_with_shuffle_is_flagged(tmp_path: Path) -> None:
    code = """
import pandas as pd
from sklearn.model_selection import KFold

df = pd.read_csv("data.csv")
cv = KFold(n_splits=5, shuffle=True)
for train_idx, test_idx in cv.split(df):
    print(len(train_idx), len(test_idx))
"""
    found = _ids_for(audit_script(_write(tmp_path, code)), "random_state")
    assert len(found) == 1, found


def test_shuffle_false_split_is_not_flagged(tmp_path: Path) -> None:
    """Near miss: an explicit time-ordered split takes no randomness."""
    code = """
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv").sort_values("date")
y = df["target"]
X = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(len(X_train), len(X_test), len(y_train), len(y_test))
"""
    assert "random_state" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_seeded_meta_estimator_covers_its_template(tmp_path: Path) -> None:
    """Near miss, found in the wild.

    The canonical decision-stump example. AdaBoost's ``_make_estimator``
    stamps its own ``random_state`` onto every clone of the template, so the
    inner ``DecisionTreeClassifier()`` is seeded even though it does not say
    so. The same prototype shape accounts for all nine hits an earlier
    version of this rule produced inside scikit-learn's own ensemble module.
    """
    code = """
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=0)
model.fit(X, y)
"""
    assert "random_state" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_pipeline_does_not_cover_its_estimator(tmp_path: Path) -> None:
    """Pipeline clones without seeding, so the inner estimator really is unseeded."""
    code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
pipe = make_pipeline(StandardScaler(), RandomForestClassifier())
pipe.fit(X, y)
"""
    found = _ids_for(audit_script(_write(tmp_path, code)), "random_state")
    assert len(found) == 1, found
    assert "RandomForestClassifier" in found[0]["message"]


def test_kmeans_with_explicit_centroids_is_silent(tmp_path: Path) -> None:
    """Near miss, found in the wild.

    Reduced from scikit-learn's ``preprocessing/_discretization.py``: KMeans
    handed explicit starting centroids is deterministic, and passing
    ``random_state`` would change nothing.
    """
    code = """
import numpy as np
from sklearn.cluster import KMeans

column = np.arange(100.0)
uniform_edges = np.linspace(column.min(), column.max(), 6)
init = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5
km = KMeans(n_clusters=5, init=init, n_init=1)
print(km.fit(column[:, None]).cluster_centers_)
"""
    assert "random_state" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_kmeans_with_strategy_name_init_is_flagged(tmp_path: Path) -> None:
    """A strategy name leaves the initialisation random."""
    code = """
import numpy as np
from sklearn.cluster import KMeans

column = np.arange(100.0).reshape(-1, 1)
km = KMeans(n_clusters=5, init="k-means++", n_init=10)
print(km.fit(column).cluster_centers_)
"""
    found = _ids_for(audit_script(_write(tmp_path, code)), "random_state")
    assert len(found) == 1, found


def test_kwargs_unpacking_suppresses_random_state(tmp_path: Path) -> None:
    """Near miss: random_state may well be inside the dict; the AST cannot tell."""
    code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

params = {"n_estimators": 300, "random_state": 0}
model = RandomForestClassifier(**params)
print(model, pd.__version__)
"""
    assert "random_state" not in _rule_ids(audit_script(_write(tmp_path, code)))


# --------------------------------------------------------------------------- #
# Rule: unused_holdout                                                         #
# --------------------------------------------------------------------------- #


def test_holdout_never_scored_warns(tmp_path: Path) -> None:
    code = """
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")
"""
    found = _ids_for(audit_script(_write(tmp_path, code)), "unused_holdout")
    assert len(found) == 1, found
    assert found[0]["severity"] == "warning"
    assert "X_test" in found[0]["message"]
    assert "y_test" in found[0]["message"]


def test_holdout_scored_is_silent(tmp_path: Path) -> None:
    code = """
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
print(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
joblib.dump(model, "model.pkl")
"""
    assert "unused_holdout" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_underscore_discards_are_not_reported_as_unused(tmp_path: Path) -> None:
    """Near miss: ``_`` is the conventional way to discard a value on purpose."""
    code = """
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)
print(len(X_train), len(y_train))
"""
    assert "unused_holdout" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_holdout_returned_from_helper_is_silent(tmp_path: Path) -> None:
    """Near miss: the names are used -- by being returned."""
    code = """
import pandas as pd
from sklearn.model_selection import train_test_split


def load(path):
    df = pd.read_csv(path)
    y = df["target"]
    X = df.drop(columns=["target"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_tr, X_te, y_tr, y_te
"""
    assert "unused_holdout" not in _rule_ids(audit_script(_write(tmp_path, code)))


# --------------------------------------------------------------------------- #
# Rule: metric_choice                                                          #
# --------------------------------------------------------------------------- #


def test_accuracy_as_only_metric_is_info(tmp_path: Path) -> None:
    code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
print(accuracy_score(y_test, model.predict(X_test)))
"""
    found = _ids_for(audit_script(_write(tmp_path, code)), "metric_choice")
    assert len(found) == 1, found
    assert found[0]["severity"] == "info"
    # The claim must be conditional -- a static read cannot see the label balance.
    assert "if " in found[0]["message"].lower()
    assert "cannot" in found[0]["message"].lower()


def test_accuracy_beside_another_metric_is_silent(tmp_path: Path) -> None:
    """Near miss: accuracy is fine when it is not the only thing reported."""
    code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
print(accuracy_score(y_test, model.predict(X_test)))
print(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
"""
    assert "metric_choice" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_classification_report_alone_is_silent(tmp_path: Path) -> None:
    """Near miss: classification_report already carries per-class precision/recall."""
    code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
"""
    assert "metric_choice" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_non_accuracy_scoring_kwarg_is_silent(tmp_path: Path) -> None:
    """Near miss: the second metric arrives as a ``scoring=`` string, not an import."""
    code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

df = pd.read_csv("data.csv")
y = df["target"]
X = df.drop(columns=["target"])
model = RandomForestClassifier(random_state=0)
print(cross_val_score(model, X, y, cv=5, scoring="roc_auc"))
print(accuracy_score(y, model.fit(X, y).predict(X)))
"""
    assert "metric_choice" not in _rule_ids(audit_script(_write(tmp_path, code)))


def test_regression_script_does_not_draw_metric_choice(tmp_path: Path) -> None:
    """Near miss: no accuracy anywhere, so the rule has no trigger."""
    code = """
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
y = df["price"]
X = df.drop(columns=["price"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = LinearRegression().fit(X_train, y_train)
print(mean_squared_error(y_test, model.predict(X_test)))
"""
    assert "metric_choice" not in _rule_ids(audit_script(_write(tmp_path, code)))


# --------------------------------------------------------------------------- #
# Regression guard: the new rules must not fire on deep-learning scripts       #
# --------------------------------------------------------------------------- #


def test_torch_script_draws_no_sklearn_rules(tmp_path: Path) -> None:
    code = """
import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Linear(10, 1)
model.train()
model.eval()
loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
train_ds, val_ds = torch.utils.data.random_split(ds, [0.8, 0.2])
"""
    ids = set(_rule_ids(audit_script(_write(tmp_path, code))))
    new_rules = {
        "preprocess_leak",
        "refit_across_split",
        "target_leak",
        "random_state",
        "unused_holdout",
        "metric_choice",
    }
    assert not (ids & new_rules), ids
