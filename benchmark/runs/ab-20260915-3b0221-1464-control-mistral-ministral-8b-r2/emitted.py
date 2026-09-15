import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump
from tqdm import tqdm

def main():
    # Define paths
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r2\train.csv"
    model_save_path = "trained_model.joblib"

    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    # Separate features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=20260915, stratify=y
    )

    # Standardize features (important for some models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Initialize and train the model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=20260915,
        n_jobs=-1  # Use all available cores
    )

    # Train with progress bar
    model.fit(X_train_scaled, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {val_accuracy:.4f}")

    # Save the model and scaler together
    print("Saving model...")
    dump({
        'model': model,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }, model_save_path)

    print(f"Model saved to {os.path.abspath(model_save_path)}")

if __name__ == "__main__":
    main()
