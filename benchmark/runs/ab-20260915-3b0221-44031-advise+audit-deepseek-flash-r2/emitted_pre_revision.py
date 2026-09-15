import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r2\train.csv"
MODEL_PATH = "model.joblib"

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["price"])
y = df["price"]

model = HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.05,
    max_iter=500,
    max_leaf_nodes=31,
    l2_regularization=0.1,
    early_stopping=False,
    random_state=42,
)

model.fit(X, y)
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
