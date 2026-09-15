import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load the dataset
df = pd.read_csv(r'C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed20260915-r2\train.csv')

# Separate features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit a model on the entire training dataset (no cross-validation here)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save the fitted model to a file in the current working directory
dump(model, 'fitted_model.joblib')
