import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed0-r1\train.csv"
MODEL_PATH = "model.joblib"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {os.path.abspath(MODEL_PATH)}")

if __name__ == "__main__":
    main()
