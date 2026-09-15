"""Score whatever the emitted script saved against the harness's holdout.

Runs as its own process: loading a file written by model-generated code is
arbitrary code execution, and a scorer that dies must not take the harness
with it. Prints one JSON object on stdout and nothing else.
"""

import json
import pickle
import sys
from pathlib import Path


def _load(path):
    try:
        import joblib

        return joblib.load(path), ""
    except Exception as exc:  # noqa: BLE001
        joblib_error = "joblib: %s: %s" % (type(exc).__name__, exc)
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle), ""
    except Exception as exc:  # noqa: BLE001
        return None, "%s; pickle: %s: %s" % (joblib_error, type(exc).__name__, exc)


def main():
    workspace, holdout_path = Path(sys.argv[1]), Path(sys.argv[2])
    target, task = sys.argv[3], sys.argv[4]
    import pandas as pd
    from sklearn.metrics import f1_score, r2_score, roc_auc_score

    patterns = ("*.joblib", "*.pkl", "*.pickle", "*.sav", "*.jbl", "*.model", "*.bin")
    candidates = sorted({p for pat in patterns for p in workspace.rglob(pat)}, key=lambda p: str(p))
    result = {"status": "unscorable", "metric": "", "score": "", "artifact": "", "attempts": []}
    if not candidates:
        result["reason"] = "the script saved no loadable model file in its workspace"
        print(json.dumps(result))
        return

    frame = pd.read_csv(holdout_path)
    if target not in frame.columns:
        result["reason"] = "holdout is missing the target column %r" % target
        print(json.dumps(result))
        return
    y = frame[target]
    raw_X = frame.drop(columns=[target])

    for path in candidates:
        name = str(path.relative_to(workspace))
        model, load_error = _load(path)
        if model is None:
            result["attempts"].append({"artifact": name, "error": load_error[:300]})
            continue
        if not hasattr(model, "predict"):
            kind = type(model).__name__
            result["attempts"].append(
                {"artifact": name, "error": "loaded object has no .predict (%s)" % kind}
            )
            continue

        X = raw_X
        expected = getattr(model, "feature_names_in_", None)
        if expected is not None:
            expected = [str(c) for c in expected]
            missing = [c for c in expected if c not in X.columns]
            if missing:
                result["attempts"].append(
                    {
                        "artifact": name,
                        "error": "model expects features the raw holdout does not contain: %s"
                        % ", ".join(missing[:8]),
                    }
                )
                continue
            X = X[expected]

        try:
            if task == "regression":
                score = float(r2_score(y, model.predict(X)))
                metric = "r2"
            elif task == "multiclass_classification":
                score = float(f1_score(y, model.predict(X), average="macro"))
                metric = "macro_f1"
            elif task == "binary_classification":
                classes = list(getattr(model, "classes_", []))
                if len(classes) != 2:
                    raise ValueError("model exposes %d classes, not 2" % len(classes))
                if not set(map(str, classes)) <= set(map(str, y.unique())):
                    raise ValueError(
                        "model classes %s do not match the holdout labels %s"
                        % (classes, sorted(y.unique().tolist()))
                    )
                positive = classes[1]
                if hasattr(model, "predict_proba"):
                    scores = model.predict_proba(X)[:, 1]
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(X)
                else:
                    raise ValueError("model exposes neither predict_proba nor decision_function")
                score = float(roc_auc_score((y == positive).astype(int), scores))
                metric = "roc_auc"
            else:
                raise ValueError("no metric is fixed for task %r" % task)
        except Exception as exc:  # noqa: BLE001
            result["attempts"].append(
                {"artifact": name, "error": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
            )
            continue

        result.update({"status": "scored", "metric": metric, "score": score, "artifact": name})
        print(json.dumps(result))
        return

    result["reason"] = "no saved artefact could be loaded and used to predict on the holdout"
    print(json.dumps(result))


main()
