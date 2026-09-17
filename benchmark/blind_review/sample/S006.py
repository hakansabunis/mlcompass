import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from joblib import dump
import os

def train_and_save_model():
    # Set random seeds for reproducibility
    random.seed(20260915)
    np.random.seed(20260915)

    # Define the path to the CSV file
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\<TMPDIR>\train.csv"

    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    # Remove duplicates to avoid inflating held-out scores
    print("Removing duplicates...")
    df = df.drop_duplicates()

    # Check and report class distribution
    class_distribution = df['Class'].value_counts(normalize=True)
    print(f"\nClass distribution:\n{class_distribution}")

    # Split features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Split into train and test sets (stratified to maintain class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=20260915, stratify=y
    )

    # Initialize and train the model
    print("\nTraining model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=20260915,
        class_weight='balanced',
        verbose=1
    )
    model.fit(X_train, y_train)

    # Evaluate on test set
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC AUC

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Print classification report for detailed metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model to a file
    model_filename = "random_forest_model.joblib"
    dump(model, model_filename)
    print(f"\nModel saved to {os.path.abspath(model_filename)}")

if __name__ == "__main__":
    train_and_save_model()
