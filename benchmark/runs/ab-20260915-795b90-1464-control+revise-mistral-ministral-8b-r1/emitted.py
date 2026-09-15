import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os

def train_and_save_model():
    # Define the file path (using forward slashes for cross-platform compatibility)
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed0-r1\train.csv"

    try:
        # Load the dataset with explicit dtype handling to ensure consistency
        df = pd.read_csv(csv_path, dtype={
            'V1': 'int64',
            'V2': 'int64',
            'V3': 'int64',
            'V4': 'int64',
            'Class': 'int64'
        })

        # Separate features and target
        X = df.drop(columns=['Class'])
        y = df['Class']

        # Split the data into training and validation sets with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Create a pipeline with classifier only (no scaling needed for tree-based models)
        # RandomForestClassifier is scale-invariant
        pipeline = Pipeline([
            ('classifier', RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                class_weight='balanced'  # Helps with potential class imbalance
            ))
        ])

        # Train the model
        pipeline.fit(X_train, y_train)

        # Evaluate on validation set
        y_pred = pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {accuracy:.4f}")

        # Save the trained model
        model_path = os.path.join(os.getcwd(), 'trained_model.joblib')
        joblib.dump(pipeline, model_path)
        print(f"Model saved to {model_path}")

    except FileNotFoundError:
        raise FileNotFoundError(f"The file at path {csv_path} was not found. Please check the path.")
    except Exception as e:
        raise RuntimeError(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    train_and_save_model()
