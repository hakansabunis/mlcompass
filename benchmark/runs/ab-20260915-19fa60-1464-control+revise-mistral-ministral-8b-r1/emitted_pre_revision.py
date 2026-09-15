import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
import os

def main():
    # File path
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed0-r1\train.csv"

    # Load data
    data = pd.read_csv(csv_path)

    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train model (RandomForest as a good default choice)
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'  # handles potential class imbalance
    )

    # Train the model
    clf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Test Accuracy: {accuracy:.4f}")

    # Save the trained model
    model_filename = "trained_model.joblib"
    dump(clf, model_filename)
    print(f"Model saved to {os.path.abspath(model_filename)}")

if __name__ == "__main__":
    main()
