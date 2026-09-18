import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from joblib import dump

# Load the dataset
file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1480-seed1-r2\train.csv"
df = pd.read_csv(file_path)

# Separate features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (optional but can improve performance for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_scaled, y_train)

# Evaluate the model (optional but good practice to ensure training was successful)
# scores = model.score(X_test_scaled, y_test)
# print(f"Model accuracy: {scores}")

# Save the trained model
dump(model, 'fitted_model.joblib')

print("Model has been saved to fitted_model.joblib")
