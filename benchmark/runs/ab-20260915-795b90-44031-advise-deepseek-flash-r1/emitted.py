import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Path to the training data
DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed0-r1\train.csv"
MODEL_PATH = "model.joblib"

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Separate features and target
target_col = "price"
X = df.drop(columns=[target_col])
y = df[target_col]

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the model
model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=1000,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=0
)

# Train on the training split
model.fit(X_train, y_train)

# Evaluate on the validation split
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
print(f"Validation RMSE: {rmse:.5f}")
print(f"Validation R2: {r2:.5f}")

# Retrain on the full dataset
final_model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=1000,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=0
)
final_model.fit(X, y)

# Save the fitted model
joblib.dump(final_model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
