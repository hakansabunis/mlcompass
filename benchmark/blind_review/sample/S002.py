import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump

# Load the dataset
file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\<TMPDIR>\train.csv"
df = pd.read_csv(file_path)

# Separate features (X) and target variable (y)
X = df.drop(columns=['price'])
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Score the held-out data
score = model.score(X_test, y_test)
print(f"Test set R^2 score: {score}")

# Save the trained model to a file in the current working directory
model_file_path = "trained_model.joblib"
dump(model, model_file_path)
print(f"Model saved to {model_file_path}")
