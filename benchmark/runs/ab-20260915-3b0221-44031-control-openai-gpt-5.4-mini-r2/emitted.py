import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------------------
# Configuration
# ----------------------------
DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r2\train.csv"
TARGET_COL = "price"
MODEL_PATH = "trained_model.joblib"

RANDOM_STATE = 42

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(DATA_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in data.")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Ensure numeric features only (as expected from the provided schema)
X = X.astype(float)
y = y.astype(float)

# ----------------------------
# Split for validation
# ----------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ----------------------------
# Model pipeline
# ----------------------------
pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
    ]
)

# Small, robust hyperparameter search
param_grid = {
    "model__max_depth": [None, 4, 6, 8],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__min_samples_leaf": [10, 20, 30],
    "model__l2_regularization": [0.0, 0.1, 1.0],
}

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=5,
    n_jobs=-1,
    verbose=1,
)

# ----------------------------
# Train
# ----------------------------
search.fit(X_train, y_train)

best_model = search.best_estimator_

# ----------------------------
# Evaluate
# ----------------------------
y_pred = best_model.predict(X_valid)

rmse = mean_squared_error(y_valid, y_pred, squared=False)
mae = mean_absolute_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

print("Best parameters:", search.best_params_)
print(f"Validation RMSE: {rmse:.6f}")
print(f"Validation MAE:  {mae:.6f}")
print(f"Validation R^2:  {r2:.6f}")

# ----------------------------
# Refit on full dataset
# ----------------------------
best_model.fit(X, y)

# ----------------------------
# Save model artifact
# ----------------------------
artifact = {
    "model": best_model,
    "feature_names": list(X.columns),
    "target": TARGET_COL,
}

joblib.dump(artifact, MODEL_PATH)
print(f"Saved trained model to: {os.path.abspath(MODEL_PATH)}")
