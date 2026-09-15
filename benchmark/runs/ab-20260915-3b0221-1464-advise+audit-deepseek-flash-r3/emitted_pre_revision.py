import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the training data
train_path = r"C:\Users\SABUNIS\AppData\Local\Temp\mlcab-1464-seed20260915-r3\train.csv"
df = pd.read_csv(train_path)

# Separate features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Define candidate models and their hyperparameter grids
models = {
    'logistic': (
        Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
        ]),
        {'clf__C': [0.1, 1, 10]}
    ),
    'random_forest': (
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        {
            'n_estimators': [100, 300, 500],
            'max_depth': [None, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4]
        }
    ),
    'gradient_boosting': (
        GradientBoostingClassifier(random_state=42),
        {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
    ),
    'hist_gradient_boosting': (
        HistGradientBoostingClassifier(random_state=42),
        {
            'max_iter': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [None, 5, 10]
        }
    )
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_score = -np.inf
best_model = None
best_name = None

# Evaluate each model
for name, (model, params) in models.items():
    grid = GridSearchCV(model, params, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model = grid.best_estimator_
        best_name = name
    print(f"{name}: {grid.best_score_:.4f}")

print(f"Best model: {best_name} with score {best_score:.4f}")

# Fit the best model on the full training data (GridSearchCV already refits, but do it again for safety)
best_model.fit(X, y)

# Save the fitted model to the current working directory
joblib.dump(best_model, 'model.joblib')
print("Model saved to model.joblib")
