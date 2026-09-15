import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os

def main():
    # Define paths
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r1\train.csv"
    model_save_path = "random_forest_model.joblib"

    # Load and prepare data
    print("Loading data...")
    df = pd.read_csv(csv_path)

    # Check for duplicates and remove them if they exist
    if df.duplicated().any():
        print(f"Found {df.duplicated().sum()} duplicate rows. Removing them...")
        df = df.drop_duplicates()

    # Split features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Split into train and test sets (stratified to maintain class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale features (important for many algorithms)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'  # handles class imbalance
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model and scaler to file
    print("\nSaving model and scaler...")
    dump({
        'model': model,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
