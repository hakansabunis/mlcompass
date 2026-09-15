import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
import os

# Define file path
csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r1\train.csv"

# Load the dataset
print("Loading dataset...")
data = pd.read_csv(csv_path)

# Separate features and target
X = data.drop(columns=['Class'])
y = data['Class']

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize and train the model
print("Training model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    max_depth=10
)
model.fit(X_train, y_train)

# Evaluate on validation set
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation accuracy: {accuracy:.4f}")

# Save the trained model
model_filename = "trained_model.joblib"
dump(model, model_filename)
print(f"Model saved to {os.path.abspath(model_filename)}")

print("Training complete!")
