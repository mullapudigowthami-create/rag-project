# Step 1 - Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

# Step 2 - Load and Explore Data
df = pd.read_csv(r'C:\Users\Admin\Downloads\creditcard.csv')

print(df.head())
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Transaction counts:\n", df['Class'].value_counts())

sns.countplot(x='Class', data=df)
plt.title('Fraud vs Genuine Transactions')
plt.xlabel('Class (0=Genuine, 1=Fraud)')
plt.ylabel('Count')
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the dataset
# Assuming creditcard.csv is in the same directory
df = pd.read_csv('creditcard.csv')

# 2. Explore the data
print(df.head())
print(df['Class'].value_counts()) # 0: Genuine, 1: Fraud

# 3. Preprocessing
# Scale 'Amount' and 'Time' (others are already PCA transformed)
scaler = StandardScaler()
df['NormalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop(['Amount', 'Time'], axis=1)

# 4. Split data
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Train Model (Random Forest is often used for this dataset)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
