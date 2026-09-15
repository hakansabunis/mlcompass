import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r1\train.csv"
df = pd.read_csv(file_path)

# Handle duplicates as advised by mlcompass
# Dropping duplicates to avoid inflation of held-out scores
df = df.drop_duplicates()

# Define features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set to check model performance (optional)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Save the trained model to the current directory
joblib_file = "trained_model.joblib"
joblib.dump(model, joblib_file)
print(f"Trained model saved as {joblib_file}")
