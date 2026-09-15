import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

# Path to the training data
train_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r3\train.csv"
target_col = "price"
model_path = "price_model.joblib"

# Load the dataset
df = pd.read_csv(train_path)

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Initialize the model
model = HistGradientBoostingRegressor(
    random_state=42,
    max_iter=500,
    learning_rate=0.05,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)

# Train the model
model.fit(X, y)

# Save the fitted model to the current working directory
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
