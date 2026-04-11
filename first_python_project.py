import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create some sample data
data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'Sales': [1500, 1800, 1200, 2100, 1900, 2400]
}

# Create a table
df = pd.DataFrame(data)

# Show the data
print("Sales Data:")
print(df)

# Show basic stats
print("\nBasic Statistics:")
print(df['Sales'].describe())

# Draw a chart
plt.plot(df['Month'], df['Sales'], marker='o', color='blue')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()