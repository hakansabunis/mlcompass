import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r2\train.csv"
MODEL_PATH = "model.joblib"

# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["price"])
y = df["price"]

# Hold out a validation set (20%) for evaluation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define model
model = HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.05,
    max_iter=500,
    max_leaf_nodes=31,
    l2_regularization=0.1,
    early_stopping=False,
    random_state=42,
)

# Train on training split and evaluate on validation split
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)
rmse = root_mean_squared_error(y_val, y_val_pred)
print(f"Validation RMSE: {rmse:.4f}")

# Refit on the full dataset for final model
final_model = HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.05,
    max_iter=500,
    max_leaf_nodes=31,
    l2_regularization=0.1,
    early_stopping=False,
    random_state=42,
)
final_model.fit(X, y)

# Save the fitted model
joblib.dump(final_model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
