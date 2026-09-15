import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed2-r3\train.csv"
df = pd.read_csv(csv_path)

feature_cols = ["V1", "V2", "V3", "V4"]
target_col = "Class"

X = df[feature_cols].to_numpy()
y = df[target_col].to_numpy()

if len(set(y)) < 2:
    from sklearn.dummy import DummyClassifier
    model = DummyClassifier(strategy="most_frequent")
else:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

model.fit(X, y)
joblib.dump(model, "model.joblib", compress=3)
print("Saved fitted model to model.joblib")
