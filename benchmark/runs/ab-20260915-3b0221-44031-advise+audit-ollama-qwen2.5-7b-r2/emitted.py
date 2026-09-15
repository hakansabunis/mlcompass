import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
file_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-44031-seed20260915-r2\train.csv"
data = pd.read_csv(file_path)

# Split into features and target
X = data.drop(columns=['price'])
y = data['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the fitted model to a file in the current working directory
joblib_file = "fitted_model.joblib"
joblib.dump(model, joblib_file)
