import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Path to the training data
csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed0-r1\train.csv"
if not os.path.exists(csv_path):
    csv_path = "train.csv"

# Load data
df = pd.read_csv(csv_path)

# Remove duplicate rows before splitting
df = df.drop_duplicates().reset_index(drop=True)

# Separate features and target
target_col = "price"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")
feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols]
y = df[target_col]

# Split into train+validation and test sets
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# Split train+validation into training and validation sets for early stopping
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=SEED
)

# Initial XGBoost model with early stopping on the validation set
model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
    early_stopping_rounds=50,
    eval_metric="rmse",
)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

best_iter = model.best_iteration
if best_iter is None:
    best_iter = 1000

# Final model trained on train+validation (test set remains untouched)
final_model = XGBRegressor(
    n_estimators=best_iter + 1,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
)
final_model.fit(X_trainval, y_trainval)

# Evaluate on the held-out test set (never used for training or early stopping)
y_pred = final_model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print(f"Test R2:   {r2:.4f}")

# Save the fitted model to the current working directory
model_path = "model.joblib"
joblib.dump(final_model, model_path)
print(f"Model saved to {os.path.abspath(model_path)}")
