import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r1\train.csv"
    df = pd.read_csv(csv_path)

    target_col = "price"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = dict(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=42,
    )

    # Validation fit to check generalization
    valid_model = HistGradientBoostingRegressor(**params)
    valid_model.fit(X_train, y_train)
    preds = valid_model.predict(X_valid)
    print(f"Validation RMSE: {root_mean_squared_error(y_valid, preds):.5f}")
    print(f"Validation R2:   {r2_score(y_valid, preds):.5f}")

    # Refit on all data for the final saved model
    final_model = HistGradientBoostingRegressor(**params)
    final_model.fit(X, y)

    joblib.dump(final_model, "model.joblib")


if __name__ == "__main__":
    main()
