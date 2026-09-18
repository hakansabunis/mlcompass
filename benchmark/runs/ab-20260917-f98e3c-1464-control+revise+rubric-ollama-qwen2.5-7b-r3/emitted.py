import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score  # This is an example of a metric that survives class imbalance
import joblib

# Load dataset
df = pd.read_csv("C:\\Users\\SABUNIS\\AppData\\Local\\Temp\\mlcab-1464-seed2-r3\\train.csv")

# Ensure no duplicate rows by setting the 'Class' column as unique identifier for dropping duplicates
if df.duplicated(subset='Class').any():
    df = df.drop_duplicates(subset='Class')

# Remove target variable from features
X = df.drop(columns=['Class'])
y = df['Class']

# Split into train and test sets with a fixed random state to ensure reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model using a fixed random seed for reproducibility
model = XGBClassifier(use_label_encoder=False, random_state=42)
model.fit(X_train, y_train)

# Save the fitted model to a file in the current working directory
joblib_file = "trained_model.joblib"
joblib.dump(model, joblib_file)

print(f"Model saved to {joblib_file}")

# Evaluate the model on the test set (data it was not trained on)
test_predictions = model.predict(X_test)
score = roc_auc_score(y_test, test_predictions)
print(f"ROC AUC score on holdout data: {score}")
