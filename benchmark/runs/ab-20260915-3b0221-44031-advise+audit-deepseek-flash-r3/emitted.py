import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Path to the training data
DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r3\train.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Separate features and target
target_col = "price"
X = df.drop(columns=[target_col])
y = df[target_col]

# Hold out a validation set
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize a robust gradient boosting regressor
model = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
)

# Train on the training split
model.fit(X_train, y_train)

# Evaluate on the held-out validation set
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print(f"Validation MAE: {mae:.4f}")
print(f"Validation R2:  {r2:.4f}")

# Refit the model on the full dataset for final deployment
final_model = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
)
final_model.fit(X, y)

# Save the fitted model to the current working directory
joblib.dump(final_model, "model.joblib")
print("Model successfully trained and saved as 'model.joblib'")
