import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Download stock data
df=yf.download('AAPL', start='2020-01-01', end='2024-01-01')
df=df.reset_index()


# See first 5 rows
print(df.head())

# Use only Close price
df = df[['Date', 'Close']]
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Create feature — days since start
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# Features and target
X = df[['Days']]
y = df['Close']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy metrics
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot actual vs predicted
plt.figure(figsize=(10,6))
plt.plot(df['Date'], df['Close'], label='Actual Price', color='blue')
plt.scatter(df['Date'].iloc[X_test.index], y_pred, label='Predicted Price', color='red', s=10)
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()