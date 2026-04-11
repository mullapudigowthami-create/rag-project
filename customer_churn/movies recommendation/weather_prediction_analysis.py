 # Step 1 - Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

# Step 2 - Create Dataset
data = {
    'Temperature': [30,25,15,20,35,28,18,22,32,27,
                   16,24,31,26,19,23,33,29,17,21],
    'Humidity':    [80,60,90,70,40,65,85,75,45,62,
                   88,72,42,68,87,73,43,66,86,74],
    'WindSpeed':   [10,15,20,12,8,14,18,11,9,13,
                   19,12,8,15,20,11,9,14,18,12],
    'Pressure':    [1010,1015,1005,1012,1020,1014,
                   1007,1011,1019,1013,1006,1012,
                   1020,1014,1006,1011,1019,1013,
                   1007,1012],
    'Weather':     ['Rainy','Cloudy','Rainy','Cloudy',
                   'Sunny','Cloudy','Rainy','Cloudy',
                   'Sunny','Cloudy','Rainy','Cloudy',
                   'Sunny','Cloudy','Rainy','Cloudy',
                   'Sunny','Cloudy','Rainy','Cloudy']
}

df = pd.DataFrame(data)
print("Dataset:")
print(df.head())
print("\nWeather Distribution:")
print(df['Weather'].value_counts())

# Step 3 - Visualize
sns.countplot(x='Weather', data=df)
plt.title('Weather Distribution')
plt.show()

# Step 4 - Prepare Data
X = df.drop('Weather', axis=1)
y = df['Weather']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 5 - Build and Train Model
model = RandomForestClassifier(n_estimators=100,
                               random_state=42)
model.fit(X_train, y_train)
print("\nModel trained successfully!")

# Step 6 - Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100, "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 7 - Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Weather Prediction - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Step 8 - Predict New Weather
new_data = np.array([[35, 40, 8, 1020]])
prediction = model.predict(new_data)
print("\nNew Weather Prediction:", prediction[0])