import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Load feature dataset
data = pd.read_csv("cleaned_audio_features.csv")  # Ensure the CSV filename is correct
X = data.drop(columns=["filename", "label"])
y = data['label']

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize data (SVM is sensitive to data scale)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Use SelectKBest to select the top 20 features
selector = SelectKBest(score_func=f_classif, k=20)  # Select the top 20 most relevant features
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# SVM hyperparameters
svm_params = {
    'C': [1, 10, 50, 100, 500],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}

# Random Forest hyperparameters
rf_params = {
    'n_estimators': [200, 300, 400, 500],  # Number of trees
    'max_depth': [15, 20, 25, 30],  # Tree depth
    'max_features': ['sqrt', 'log2']
}

# XGBoost hyperparameters
xgb_params = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10]
}

# Perform hyperparameter tuning using GridSearchCV
print("Optimizing SVM...")
svm_grid = GridSearchCV(SVC(), svm_params, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid.fit(X_train, y_train)

print("Optimizing Random Forest...")
rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print("Optimizing XGBoost...")
xgb_grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_params, cv=5, scoring='accuracy', n_jobs=-1)
xgb_grid.fit(X_train, y_train)

# Output best hyperparameters and accuracy
print("\nFinal Results")
print(f"SVM - Best Parameters: {svm_grid.best_params_}")
print(f"SVM - Best Accuracy: {svm_grid.best_score_:.4f}")

print(f"\nRandom Forest - Best Parameters: {rf_grid.best_params_}")
print(f"Random Forest - Best Accuracy: {rf_grid.best_score_:.4f}")

print(f"\nXGBoost - Best Parameters: {xgb_grid.best_params_}")
print(f"XGBoost - Best Accuracy: {xgb_grid.best_score_:.4f}")

# Identify the best model
best_model = max(
    [("SVM", svm_grid.best_score_), ("Random Forest", rf_grid.best_score_), ("XGBoost", xgb_grid.best_score_)],
    key=lambda x: x[1]
)

print(f"\nBest Model: {best_model[0]}, Accuracy: {best_model[1]:.4f}")
