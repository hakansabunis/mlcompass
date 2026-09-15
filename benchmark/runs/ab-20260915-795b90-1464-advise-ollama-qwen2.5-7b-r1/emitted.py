import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv(r'C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed0-r1\train.csv')

# Handle duplicate rows as advised by mlcompass
df = df.drop_duplicates()

# Split the data into features (X) and target (y)
X = df[['V1', 'V2', 'V3', 'V4']]
y = df['Class']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model to a file in the current working directory
joblib_file_path = 'trained_model.pkl'
joblib.dump(model, joblib_file_path)
print(f"Trained model saved to {joblib_file_path}")
