import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

train_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r3\train.csv"

df = pd.read_csv(train_path)
X = df.drop(columns=["Class"]).values
y = df["Class"].values

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

pipeline.fit(X, y)

model_path = os.path.join(os.getcwd(), "model.joblib")
joblib.dump(pipeline, model_path)

print(f"Model trained and saved to {model_path}")
