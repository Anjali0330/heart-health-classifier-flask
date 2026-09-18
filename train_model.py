import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import json

df = pd.read_csv('heart.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:,1])
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    results[name] = {'test_accuracy': acc, 'test_auc': auc, 'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()}
    print(f"{name:20s} | Test Acc: {acc:.4f} | AUC: {auc:.4f} | CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

best_name = max(results, key=lambda k: results[k]['test_accuracy'])
print(f"\nBest model: {best_name}")

best_model = models[best_name]
preds = best_model.predict(X_test_scaled)
print("\nClassification report for best model:")
print(classification_report(y_test, preds))
print("Confusion matrix:")
print(confusion_matrix(y_test, preds))

# Save best model + scaler for the Flask app
joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
with open('model_results.json', 'w') as f:
    json.dump({'best_model': best_name, 'results': results,
               'feature_order': X.columns.tolist()}, f, indent=2)

print("\nSaved model.pkl, scaler.pkl, model_results.json")
