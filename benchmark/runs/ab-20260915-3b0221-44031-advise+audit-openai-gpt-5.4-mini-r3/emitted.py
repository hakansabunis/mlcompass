import os
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def main():
    train_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r3\train.csv"
    target_col = "price"
    model_path = "trained_model.joblib"

    df = pd.read_csv(train_path)

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(float)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
    )

    target_transformer = FunctionTransformer(
        func=np.log1p,
        inverse_func=np.expm1,
        validate=False,
    )

    base_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=600,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=0.1,
        max_bins=255,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
    )

    model = TransformedTargetRegressor(
        regressor=Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", base_model),
            ]
        ),
        transformer=target_transformer,
    )

    model.fit(X_train, y_train)

    valid_pred = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, valid_pred, squared=False)
    print(f"Holdout RMSE: {rmse:.6f}")

    # Optional cross-validation on the full dataset for a more stable estimate.
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    print(f"CV RMSE: {-cv_scores.mean():.6f} +/- {cv_scores.std():.6f}")

    # Refit on full data before saving for maximum training signal.
    model.fit(X, y)

    joblib.dump(
        {
            "model": model,
            "features": list(X.columns),
            "target": target_col,
            "seed": SEED,
        },
        model_path,
    )

    print(f"Saved model to: {os.path.abspath(model_path)}")


if __name__ == "__main__":
    main()
