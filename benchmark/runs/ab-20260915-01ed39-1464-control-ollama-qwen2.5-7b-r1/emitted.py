import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r1\train.csv"
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop(columns=["Class"])
y = data["Class"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the model to file in the current working directory
joblib_file = "trained_model.joblib"
joblib.dump(model, joblib_file)
print(f"Model saved to {joblib_file}")
