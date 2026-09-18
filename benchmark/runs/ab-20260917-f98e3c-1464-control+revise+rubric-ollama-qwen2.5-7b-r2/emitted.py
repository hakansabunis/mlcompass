import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
file_path = r'C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed1-r2\train.csv'
df = pd.read_csv(file_path)

# Remove duplicate rows if any
df.drop_duplicates(inplace=True)

# Separate features and target
features = df.drop(columns='Class')
target = df['Class']

# Encode categorical variables if any (all are int64, no encoding needed)
# In this case, all columns are int64, so we don't need to encode them

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the model
model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file in the current working directory
joblib_file_path = 'trained_model.joblib'
joblib.dump(model, joblib_file_path)
print(f"Model saved to {joblib_file_path}")

# Evaluate the model on the test set
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy}")
