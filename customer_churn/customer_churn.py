# =============================
# Customer Churn Prediction
# =============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1 - Create Dataset
data = {
    'Age':            [25,45,35,50,23,40,60,29,48,33],
    'Monthly_Bill':   [500,1200,800,1500,300,900,1100,600,1300,700],
    'Num_Products':   [1,3,2,4,1,2,3,1,4,2],
    'Years_Customer': [1,5,3,7,1,4,6,2,8,3],
    'Churn':          [0,1,0,1,0,0,1,0,1,0]
}

df = pd.DataFrame(data)

# STEP 2 - Features and Label
X = df[['Age','Monthly_Bill','Num_Products','Years_Customer']]
y = df['Churn']

# STEP 3 - Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# STEP 4 - Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("✅ Model Trained Successfully!")

# STEP 5 - Predict
y_pred = model.predict(X_test)

# STEP 6 - Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {accuracy*100:.2f}%")

# STEP 7 - Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# STEP 8 - Predict New Customer
new_customer = [[35, 800, 2, 3]]
prediction = model.predict(new_customer)
if prediction[0] == 1:
    print("⚠️ Customer will LEAVE!")
else:
    print("✅ Customer will STAY!")