import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load the dataset
file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r3\train.csv"
df = pd.read_csv(file_path)

# Handle warnings about exact duplicate rows
df.drop_duplicates(inplace=True)

# Define features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Save the trained model to the current working directory
model_file_path = 'trained_model.joblib'
dump(clf, model_file_path)

print(f"Model saved to {model_file_path}")
