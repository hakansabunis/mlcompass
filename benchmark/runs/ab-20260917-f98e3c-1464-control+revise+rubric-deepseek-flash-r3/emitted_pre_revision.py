import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

# Path to the training CSV file
train_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed2-r3\train.csv"

# Load the dataset
df = pd.read_csv(train_path)

# Separate features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Determine if the task is classification or regression based on the number of unique target values
if y.nunique() > 20:
    # Regression
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    best_iter = model.best_iteration
    final_model = XGBRegressor(
        n_estimators=best_iter + 1,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    final_model.fit(X, y)
else:
    # Classification
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    best_iter = model.best_iteration
    final_model = XGBClassifier(
        n_estimators=best_iter + 1,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    final_model.fit(X, y)

# Save the trained model to the current working directory
joblib.dump(final_model, 'model.joblib')
