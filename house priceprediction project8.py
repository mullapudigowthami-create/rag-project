# House Price Prediction using AI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# House Data
data = {
    'Size_SqFt':    [1000,1500,1200,1800,2000,
                     2200,1900,2100,2300,2500,
                     1100,1600,1300,1900,2100],
    'Bedrooms':     [2,3,2,4,4,5,3,4,4,5,
                     2,3,3,4,4],
    'Bathrooms':    [1,2,1,2,3,3,2,2,3,3,
                     1,2,2,2,3],
    'Age_Years':    [10,5,8,3,2,1,6,4,3,1,
                     12,7,9,5,3],
    'Distance_KM':  [5,3,8,2,1,4,6,3,2,1,
                     9,4,7,3,2],
    'Price_Lakhs':  [45,75,55,95,110,130,
                     85,105,120,145,
                     40,80,60,100,115]
}

# Create DataFrame
df = pd.DataFrame(data)
print("House Data:")
print(df.head())
print(f"\nTotal Houses: {len(df)}")

# Features and Target
X = df[['Size_SqFt','Bedrooms',
        'Bathrooms','Age_Years',
        'Distance_KM']]
y = df['Price_Lakhs']

# Split Data
X_train,X_test,y_train,y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\nTraining Data: {len(X_train)} houses")
print(f"Testing Data: {len(X_test)} houses")

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
error = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Error: {error:.2f} Lakhs")
print(f"Model Accuracy: {r2*100:.2f}%")

# Predict new house
new_house = pd.DataFrame({
    'Size_SqFt':   [1800],
    'Bedrooms':    [3],
    'Bathrooms':   [2],
    'Age_Years':   [4],
    'Distance_KM': [3]
})
predicted_price = model.predict(new_house)
print(f"\nNew House Prediction:")
print(f"Size: 1800 SqFt")
print(f"Bedrooms: 3, Bathrooms: 2")
print(f"Age: 4 years, Distance: 3 KM")
print(f"Predicted Price: "
      f"{predicted_price[0]:.2f} Lakhs")

# Plots
plt.figure(figsize=(15,10))

# Plot 1 — Actual vs Predicted
plt.subplot(2,2,1)
plt.scatter(y_test, y_pred,
            color='blue',
            label='Predicted')
plt.plot([40,150],[40,150],
         color='red',
         linestyle='dashed',
         label='Perfect Prediction')
plt.xlabel('Actual Price (Lakhs)')
plt.ylabel('Predicted Price (Lakhs)')
plt.title('Actual vs Predicted Price')
plt.legend()

# Plot 2 — Size vs Price
plt.subplot(2,2,2)
plt.scatter(df['Size_SqFt'],
            df['Price_Lakhs'],
            color='green')
plt.xlabel('Size (SqFt)')
plt.ylabel('Price (Lakhs)')
plt.title('Size vs Price')

# Plot 3 — Bedrooms vs Price
plt.subplot(2,2,3)
bedroom_price = df.groupby('Bedrooms')[
    'Price_Lakhs'].mean()
plt.bar(bedroom_price.index,
        bedroom_price.values,
        color='orange')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Average Price (Lakhs)')
plt.title('Bedrooms vs Average Price')

# Plot 4 — Distance vs Price
plt.subplot(2,2,4)
plt.scatter(df['Distance_KM'],
            df['Price_Lakhs'],
            color='purple')
plt.xlabel('Distance from City (KM)')
plt.ylabel('Price (Lakhs)')
plt.title('Distance vs Price')

plt.tight_layout()
plt.show()
