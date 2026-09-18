import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump
import os
from sklearn.utils import resample

def main():
    # Define file path
    file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed1-r2\train.csv"

    # Load the dataset
    print("Loading dataset...")
    data = pd.read_csv(file_path)

    # Check for and remove duplicate rows
    print(f"Original shape: {data.shape}")
    data = data.drop_duplicates().reset_index(drop=True)
    print(f"Shape after removing duplicates: {data.shape}")

    # Separate features and target
    X = data.drop(columns=['Class'])
    y = data['Class']

    # Check class distribution
    class_counts = y.value_counts()
    print("\nClass distribution:")
    print(class_counts)

    # Handle class imbalance by resampling if needed
    if len(class_counts) > 1 and class_counts.max() / class_counts.min() > 1.5:
        print("Class imbalance detected, applying oversampling for minority class")
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()

        # Separate majority and minority classes
        df_majority = data[data['Class'] == majority_class]
        df_minority = data[data['Class'] == minority_class]

        # Upsample minority class
        df_minority_upsampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=42
        )

        # Combine majority class with upsampled minority class
        data = pd.concat([df_majority, df_minority_upsampled])

        # Re-separate features and target after resampling
        X = data.drop(columns=['Class'])
        y = data['Class']

    # Split data into training and validation sets (stratified to maintain class distribution)
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train the model with class weights
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        max_depth=10,
        min_samples_split=5
    )
    model.fit(X_train, y_train)

    # Evaluate on validation set
    print("\nValidation results:")
    y_pred = model.predict(X_val)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # Calculate ROC AUC if there are only two classes
    if len(class_counts) == 2:
        y_proba = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_proba)
        print(f"ROC AUC Score: {roc_auc:.4f}")
    else:
        print("Multiple classes detected - ROC AUC not calculated")

    # Save the model to file
    model_file = os.path.join(os.getcwd(), "mlcab_model.joblib")
    dump(model, model_file)
    print(f"\nModel saved to {model_file}")

if __name__ == "__main__":
    main()
