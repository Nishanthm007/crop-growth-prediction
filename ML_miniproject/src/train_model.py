# src/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from sklearn.metrics import classification_report

# Load dataset from correct relative path
df = pd.read_csv('../data/Crop_recommendation.csv')

# Features and label
X = df.drop('label', axis=1)
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, '../model/crop_model.pkl')
print("✅ Model saved to: ../model/crop_model.pkl")

# ✅ Save feature importances for explainability
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

importance_df.to_csv('../model/feature_importance.csv', index=False)
print("✅ Feature importances saved to ../model/feature_importance.csv")