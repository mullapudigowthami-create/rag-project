import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Past sales data
months = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
sales = np.array([1500, 1800, 1200, 2100, 1900, 2400])

# Create AI model
model = LinearRegression()
model.fit(months, sales)

# Predict next 3 months
future_months = np.array([7, 8, 9]).reshape(-1, 1)
predictions = model.predict(future_months)

print("Predicted Sales:")
print(f"Month 7: {predictions[0]:.0f}")
print(f"Month 8: {predictions[1]:.0f}")
print(f"Month 9: {predictions[2]:.0f}")

# Plot everything
plt.plot([1,2,3,4,5,6], sales, marker='o', color='blue', label='Actual Sales')
plt.plot([7,8,9], predictions, marker='o', color='red', linestyle='--', label='AI Predicted')
plt.title('Sales Prediction using AI')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()