import os
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_absolute_error


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def main():
    train_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed0-r1\train.csv"
    output_path = os.path.join(os.getcwd(), "trained_model.joblib")

    df = pd.read_csv(train_path)

    target_col = "price"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(float)

    # Explicit holdout split for a quick validation check.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=SEED,
        shuffle=True,
    )

    base_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=None,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=1e-4,
        max_iter=600,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
    )

    regressor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", base_model),
        ]
    )

    model = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )

    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    print(f"Holdout MAE: {val_mae:.6f}")

    # Also compute a more stable cross-validated estimate on the full dataset.
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=mae_scorer, n_jobs=-1)
    print(f"5-fold CV MAE: {-cv_scores.mean():.6f} ± {cv_scores.std():.6f}")

    # Refit on all data before saving the final model.
    model.fit(X, y)

    payload = {
        "model": model,
        "features": list(X.columns),
        "target": target_col,
        "seed": SEED,
        "holdout_mae": float(val_mae),
        "cv_mae_mean": float(-cv_scores.mean()),
        "cv_mae_std": float(cv_scores.std()),
    }
    joblib.dump(payload, output_path)
    print(f"Saved trained model to: {output_path}")


if __name__ == "__main__":
    main()
