import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import dump
from tqdm import tqdm

def train_and_save_model():
    # Load the dataset
    file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r2\train.csv"
    df = pd.read_csv(file_path)

    # Separate features and target
    X = df.drop(columns=['price'])
    y = df['price']

    # Handle potential outliers by capping extreme values (based on IQR analysis)
    # We'll use 99th percentile as upper bound for features with outliers
    for col in X.columns:
        upper_bound = X[col].quantile(0.99)
        X[col] = X[col].clip(upper=upper_bound)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=20260915
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train Gradient Boosting Regressor
    # Using early stopping based on validation performance
    model = GradientBoostingRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        random_state=20260915,
        verbose=1
    )

    # Track validation performance
    best_score = float('inf')
    patience = 20
    no_improvement_count = 0

    for _ in tqdm(range(100), desc="Training with early stopping"):
        model.fit(X_train_scaled, y_train)

        # Evaluate on validation set
        val_preds = model.predict(X_val_scaled)
        current_score = mean_squared_error(y_val, val_preds, squared=False)

        if current_score < best_score:
            best_score = current_score
            no_improvement_count = 0
            best_model = model.copy()
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            break

    # Save the best model and preprocessing objects
    model_to_save = {
        'model': best_model,
        'scaler': scaler,
        'feature_columns': X.columns.tolist()
    }

    dump(model_to_save, 'house_price_model.joblib')
    print(f"Model saved to house_price_model.joblib")
    print(f"Best validation RMSE: {best_score:.4f}")

if __name__ == "__main__":
    train_and_save_model()
