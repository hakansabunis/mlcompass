import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load the dataset
file_path = r'C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r3\train.csv'
df = pd.read_csv(file_path)

# Separate features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model and fit on the training data
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the fitted model to a file in the current working directory
dump(model, 'fitted_model.joblib')
