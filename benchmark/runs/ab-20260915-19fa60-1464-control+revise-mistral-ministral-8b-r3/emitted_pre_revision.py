import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
import os

# Define the path to the CSV file
csv_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed2-r3\train.csv"

# Load the dataset
print("Loading dataset...")
data = pd.read_csv(csv_path)

# Separate features and target
X = data.drop(columns=['Class'])
y = data['Class']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation accuracy: {accuracy:.4f}")

# Save the trained model to a file
model_path = os.path.join(os.getcwd(), 'random_forest_model.joblib')
print(f"Saving model to {model_path}...")
dump(model, model_path)
print("Training complete.")
