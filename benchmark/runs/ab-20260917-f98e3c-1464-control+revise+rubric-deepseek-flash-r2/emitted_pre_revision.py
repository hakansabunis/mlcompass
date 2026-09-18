import os
import pandas as pd
import joblib
from xgboost import XGBClassifier

csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed1-r2\train.csv"
df = pd.read_csv(csv_path)

target_col = "Class"
X = df.drop(columns=[target_col])
y = df[target_col]

n_classes = y.nunique()
eval_metric = "logloss" if n_classes == 2 else "mlogloss"

model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    eval_metric=eval_metric,
)

model.fit(X, y)

model_path = os.path.join(os.getcwd(), "trained_model.joblib")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
