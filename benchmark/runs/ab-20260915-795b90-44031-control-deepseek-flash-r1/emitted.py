import os
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# Path to the training CSV
csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed0-r1\train.csv"
target_column = "price"

# Load data
df = pd.read_csv(csv_path)

# Separate features and target
X = df.drop(columns=[target_column]).to_numpy()
y = df[target_column].to_numpy()

# Initialize and train the model
model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    random_state=42
)
model.fit(X, y)

# Save the fitted model to the current working directory
model_path = os.path.join(os.getcwd(), "model.joblib")
joblib.dump(model, model_path)
