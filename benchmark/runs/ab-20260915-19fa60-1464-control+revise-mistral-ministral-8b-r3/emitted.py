import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    try:
        # Define the path to the CSV file (using raw string for Windows path)
        csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed2-r3\train.csv"

        # Verify file exists before loading
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found at path: {csv_path}")

        print("Loading dataset...")
        data = pd.read_csv(csv_path)

        # Verify required columns exist
        required_columns = ['V1', 'V2', 'V3', 'V4', 'Class']
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            raise ValueError(f"Missing required columns: {missing}")

        # Separate features and target
        X = data[['V1', 'V2', 'V3', 'V4']]  # Explicitly specify feature columns
        y = data['Class']

        # Split the data (stratified split for classification)
        print("Splitting data...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Initialize and train the model with better parameters
        print("Training model...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,  # Use all available cores
            class_weight='balanced'  # Handle potential class imbalance
        )
        model.fit(X_train, y_train)

        # Evaluate the model
        print("Evaluating model...")
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))

        # Save the trained model to a file in current directory
        model_filename = 'rf_classifier.joblib'
        model_path = os.path.join(os.getcwd(), model_filename)
        print(f"\nSaving model to: {model_path}")
        dump(model, model_path)
        print("Training completed successfully!")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
