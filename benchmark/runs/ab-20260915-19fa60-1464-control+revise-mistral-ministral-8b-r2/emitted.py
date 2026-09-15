import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import os
from tqdm import tqdm

def train_and_save_model():
    try:
        # Load the dataset with error handling
        csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed1-r2\train.csv"

        # Check if file exists before loading
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"The file {csv_path} does not exist")

        print("Loading data...")
        data = pd.read_csv(csv_path)

        # Verify expected columns exist
        expected_columns = ['V1', 'V2', 'V3', 'V4', 'Class']
        missing_cols = [col for col in expected_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Separate features and target
        X = data[['V1', 'V2', 'V3', 'V4']]
        y = data['Class']

        # Split data into training and validation sets
        print("Splitting data...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y,
            shuffle=True
        )

        # Initialize and train the model with progress bar
        print("Training model...")
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1  # Use all available cores
        )

        # Train with progress tracking
        model.fit(X_train, y_train)

        # Evaluate on validation set
        print("Evaluating model...")
        val_predictions = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)

        print(f"\nValidation Accuracy: {val_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, val_predictions))

        # Create output directory if it doesn't exist
        os.makedirs(os.getcwd(), exist_ok=True)
        model_filename = os.path.join(os.getcwd(), 'random_forest_model.joblib')

        # Save the trained model
        print("Saving model...")
        dump(model, model_filename, compress=3)
        print(f"Model successfully saved to {model_filename}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model()
