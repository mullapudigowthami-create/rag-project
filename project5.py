import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Student Data
data = {
    'Study_Hours': [2,3,4,5,6,7,8,9,10,1,
                    3,5,7,9,2,4,6,8,10,1],
    'Attendance':  [60,65,70,75,80,85,90,95,100,50,
                    65,75,85,95,60,70,80,90,100,55],
    'Previous_Score':[40,50,55,60,65,70,75,80,90,30,
                      50,60,70,85,45,55,65,75,95,35],
    'Final_Grade': [45,52,58,65,70,75,80,85,95,35,
                    53,63,73,88,48,58,68,78,97,38]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and Target
X = df[['Study_Hours', 'Attendance', 'Previous_Score']]
y = df['Final_Grade']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
error = mean_absolute_error(y_test, y_pred)
print(f"Model Error: {error:.2f}")

# Predict new student
new_student = pd.DataFrame({
    'Study_Hours': [6],
    'Attendance': [80],
    'Previous_Score': [65]
})
predicted_grade = model.predict(new_student)
print(f"Predicted Grade: {predicted_grade[0]:.2f}")

# Plot Graph
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted')
plt.plot([30,100],[30,100], color='red', 
         linestyle='dashed', label='Perfect Prediction')
plt.xlabel('Actual Grades')
plt.ylabel('Predicted Grades')
plt.title('Student Grade Prediction using AI')
plt.legend()
plt.show()
