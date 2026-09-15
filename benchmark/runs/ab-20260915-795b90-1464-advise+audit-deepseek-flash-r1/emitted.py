import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed0-r1\train.csv"
target_col = "Class"
model_path = os.path.join(os.getcwd(), "model.joblib")

df = pd.read_csv(csv_path)

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# Validation split for auditing and sanity check
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )),
])

# Evaluate on the hold-out split
model.fit(X_train, y_train)
val_score = model.score(X_val, y_val)
print(f"Validation accuracy: {val_score:.4f}")

# Retrain on the full dataset for the final saved model
model.fit(X, y)
joblib.dump(model, model_path)
