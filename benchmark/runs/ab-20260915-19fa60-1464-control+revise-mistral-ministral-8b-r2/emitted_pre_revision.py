import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
import os

def train_and_save_model():
    # Load the dataset
    csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed1-r2\train.csv"
    data = pd.read_csv(csv_path)

    # Separate features and target
    X = data.drop(columns=['Class'])
    y = data['Class']

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # Evaluate on validation set
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Save the trained model
    model_filename = os.path.join(os.getcwd(), 'random_forest_model.joblib')
    dump(model, model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    train_and_save_model()
