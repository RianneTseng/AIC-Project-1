import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('cleaned_audio_features.csv')
X = df.drop(columns=['label'])  # Features
y = df['label']  # Target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameters
svm_params = {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}
rf_params = {'n_estimators': 300, 'max_depth': 20, 'max_features': 'sqrt'}
xgb_params = {'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.1, 'eval_metric': 'logloss'}

# Train SVM
svm_model = SVC(**svm_params)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
print("SVM Performance:")
print(classification_report(y_test, svm_pred))
print("Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, svm_pred), annot=True, fmt='d')
plt.title("SVM Confusion Matrix")
plt.savefig("svm_confusion_matrix.png")
plt.close()

# Train Random Forest
rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest Performance:")
print(classification_report(y_test, rf_pred))
print("Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d')
plt.title("Random Forest Confusion Matrix")
plt.savefig("rf_confusion_matrix.png")
plt.close()

# Train XGBoost
xgb_model = XGBClassifier(**xgb_params)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print("XGBoost Performance:")
print(classification_report(y_test, xgb_pred))
print("Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, xgb_pred), annot=True, fmt='d')
plt.title("XGBoost Confusion Matrix")
plt.savefig("xgb_confusion_matrix.png")
plt.close()

# K-Means Clustering with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_pca)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.5)
plt.title("K-Means Clustering (PCA-reduced features)")
plt.savefig("kmeans_pca_plot.png")
plt.close()

print("Training complete. Results saved.")
