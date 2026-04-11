# ============================================
# CREDIT CARD FRAUD DETECTION - COMPLETE CODE
# ============================================

# Step 1 - Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

# Step 2 - Load Data
df = pd.read_csv('creditcard.csv')
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Transaction counts:\n", df['Class'].value_counts())

# Step 3 - Visualize Fraud vs Genuine
sns.countplot(x='Class', data=df)
plt.title('Fraud vs Genuine Transactions')
plt.xlabel('Class (0=Genuine, 1=Fraud)')
plt.ylabel('Count')
plt.show()

# Step 4 - Prepare Data
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Step 5 - Build and Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model trained successfully!")

# Step 6 - Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100, "%")
print("Classification Report:\n", 
      classification_report(y_test, y_pred))

# Step 7 - Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Genuine', 'Fraud'],
            yticklabels=['Genuine', 'Fraud'])
plt.title('Credit Card Fraud Detection')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Step 8 - Predict New Transaction
new_transaction = np.array([df.drop('Class', 
                            axis=1).iloc[0]])
result = model.predict(new_transaction)
if result[0] == 1:
    print("⚠️ Fraudulent Transaction Detected!")
else:
    print("✅ Genuine Transaction")