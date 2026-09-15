import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r3\train.csv"
MODEL_PATH = "trained_model.joblib"


def main():
    df = pd.read_csv(DATA_PATH)

    target_col = "price"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Hold-out split for a lightweight validation signal
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Robust, fast tree-based model that works well on tabular regression
    base_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.05,
                    max_depth=None,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    l2_regularization=0.0,
                    random_state=42,
                ),
            ),
        ]
    )

    # Optional target transform can help stabilize regression on skewed targets
    model = TransformedTargetRegressor(
        regressor=base_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )

    param_grid = {
        "regressor__model__learning_rate": [0.03, 0.05, 0.1],
        "regressor__model__max_leaf_nodes": [15, 31, 63],
        "regressor__model__min_samples_leaf": [10, 20, 40],
        "regressor__model__l2_regularization": [0.0, 0.1, 1.0],
    }

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # Validate
    y_pred = best_model.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_pred, squared=False)
    r2 = r2_score(y_valid, y_pred)

    # Refit on full data using best hyperparameters
    best_model.fit(X, y)

    # Save fitted model
    joblib.dump(
        {
            "model": best_model,
            "feature_names": list(X.columns),
            "target_col": target_col,
            "validation_rmse": float(rmse),
            "validation_r2": float(r2),
            "best_params": search.best_params_,
        },
        MODEL_PATH,
    )

    print(f"Saved model to: {os.path.abspath(MODEL_PATH)}")
    print(f"Validation RMSE: {rmse:.6f}")
    print(f"Validation R2: {r2:.6f}")
    print(f"Best params: {json.dumps(search.best_params_)}")


if __name__ == "__main__":
    main()
