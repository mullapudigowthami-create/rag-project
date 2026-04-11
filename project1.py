# Project 1 - Data Science Scatter Plot
# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create ssome sample data
data={
    'Months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'Sales': [1500, 1800, 1200, 2100, 1900, 2400]
}
#create a table
df=pd.DataFrame(data)

# Print the data table in terminal
print("Sales Data")
print(df)

#show basic stats
print("\n Basic Statistics:")
print(df['Sales'].describe())


# Draw a chart 
plt.plot(df['Months'],df['Sales'],marker='o',color='blue')
plt.title('Monthly Sales')
plt.xlabel('Months')
plt.ylabel('Sales')

# Show the graph
plt.show()


