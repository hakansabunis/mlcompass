import pandas as pd
import numpy as np
import random
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, root_mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Path to the training CSV file
train_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed2-r3\train.csv"

# Load the dataset
df = pd.read_csv(train_path)

# Remove duplicate rows to prevent data leakage
df = df.drop_duplicates()

# Separate features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Determine if the task is classification or regression
# Heuristic: if the target has few unique values, treat as classification
is_classification = y.nunique() <= 20

# Split into train and test sets (test set is held out for final evaluation)
if is_classification:
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
else:
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

# Further split training data into train and validation for early stopping
if is_classification:
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=SEED, stratify=y_train_full
    )
else:
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=SEED
    )

# Train model with early stopping
if is_classification:
    model = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    best_iter = model.best_iteration
    
    # Retrain on the full training data (train + validation) using the best iteration
    final_model = XGBClassifier(
        n_estimators=best_iter + 1,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0
    )
    final_model.fit(X_train_full, y_train_full)
    
    # Evaluate on the held-out test set
    y_pred = final_model.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy on test set: {bal_acc:.4f}")
else:
    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    best_iter = model.best_iteration
    
    final_model = XGBRegressor(
        n_estimators=best_iter + 1,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0
    )
    final_model.fit(X_train_full, y_train_full)
    
    y_pred = final_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE on test set: {rmse:.4f}")
    print(f"R2 on test set: {r2:.4f}")

# Save the final model to the current working directory
joblib.dump(final_model, 'model.joblib')
