import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
file_path = r'C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r1\train.csv'
df = pd.read_csv(file_path)

# Drop duplicate rows as advised by mlcompass
df.drop_duplicates(inplace=True)

# Separate features and target
X = df.iloc[:, :-1]
y = df['Class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set to check performance (not needed for final save)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the fitted model to disk
joblib_file_path = 'fitted_model.pkl'
joblib.dump(model, joblib_file_path)

print(f"Model saved to {joblib_file_path}")
