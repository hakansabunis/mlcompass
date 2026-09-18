import os
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.dummy import DummyClassifier

csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed0-r1\train.csv"
target_col = "Class"

df = pd.read_csv(csv_path)

if target_col not in df.columns:
    raise ValueError(f"Target column {target_col!r} not found in {csv_path}")

df = df.dropna(subset=[target_col])

X = df.drop(columns=[target_col])
y = df[target_col]

# Decide whether this is a classification or regression task
if y.dtype == object or str(y.dtype) == "category" or y.dtype == bool:
    is_classification = True
else:
    is_classification = y.nunique(dropna=True) <= 20

if is_classification:
    if y.nunique(dropna=True) <= 1:
        model = DummyClassifier(strategy="most_frequent", random_state=42)
    else:
        model = HistGradientBoostingClassifier(random_state=42)
else:
    model = HistGradientBoostingRegressor(random_state=42)

model.fit(X, y)

model_path = os.path.join(os.getcwd(), "trained_model.joblib")
joblib.dump(model, model_path)

print(f"Model saved to {model_path}")
