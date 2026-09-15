import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump

# Load the dataset
file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r1\train.csv"
df = pd.read_csv(file_path)

# Remove duplicate rows as advised by mlcompass
df = df.drop_duplicates()

# Split the data into features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model (optional)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model
dump(model, 'trained_model.joblib')
