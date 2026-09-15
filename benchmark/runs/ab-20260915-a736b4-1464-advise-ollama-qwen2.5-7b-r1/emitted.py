import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load the dataset
file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r1\train.csv"
df = pd.read_csv(file_path)

# Split into features and target variable
X = df.drop(columns=['Class'])
y = df['Class']

# Handle duplicates as suggested by mlcompass (assuming they are genuine repeats)
df = df.drop_duplicates()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set to check performance (not strictly necessary for saving the model)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model to the current working directory
joblib_file = "trained_model.joblib"
joblib.dump(model, joblib_file)
