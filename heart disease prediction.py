import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv('heart.csv')

# See first 5 rows
print(df.head())

# Check shape
print(df.shape)

# Check for missing values
print(df.isnull().sum())

# Basic statistics
print(df.describe())

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Check shapes
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

print("Model trained successfully!")
print("Predictions:", y_pred)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100, "%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# Visualize Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Heart Disease'],
            yticklabels=['No Disease', 'Heart Disease'])
plt.title('Heart Disease Prediction - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Get feature importance
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

# Sort by importance
feature_importance = feature_importance.sort_values(ascending=False)

# Visualize
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance.values, 
            y=feature_importance.index,
            palette='Blues_r')
plt.title('Heart Disease - Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

print("Most important feature:", feature_importance.index[0])
print("Feature importance scores:\n", feature_importance)


import numpy as np

# New patient data
# [age, sex, cp, trestbps, chol, fbs, restecg, 
#  thalach, exang, oldpeak, slope, ca, thal]

new_patient = np.array([[55, 1, 2, 130, 250, 0, 1, 
                         165, 0, 0.5, 2, 0, 2]])

# Predict
result = model.predict(new_patient)

if result[0] == 1:
    print("⚠️ Heart Disease Detected")
else:
    print("✅ No Heart Disease Detected")

# Prediction probability
probability = model.predict_proba(new_patient)
print("Probability of No Disease:", 
      round(probability[0][0]*100, 2), "%")
print("Probability of Heart Disease:", 
      round(probability[0][1]*100, 2), "%")