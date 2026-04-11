# Iris Flower Classification
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load famous Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, 
                  columns=iris.feature_names)
df['flower'] = iris.target

print("Iris Dataset:")
print(df.head())

# Features and Target
X = df[iris.feature_names]
y = df['flower']

# Split Data
X_train,X_test,y_train,y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")

# Predict new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)
flower_names = ['Setosa','Versicolor','Virginica']
print(f"Predicted Flower: "
      f"{flower_names[prediction[0]]}")

# Plot
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
colors = ['red','green','blue']
for i in range(3):
    plt.scatter(
        df[df['flower']==i]['sepal length (cm)'],
        df[df['flower']==i]['sepal width (cm)'],
        color=colors[i],
        label=flower_names[i]
    )
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Flower Classification')
plt.legend()

plt.subplot(1,2,2)
counts = df['flower'].value_counts()
plt.bar(flower_names, counts.values,
        color=['red','green','blue'])
plt.title('Flower Distribution')
plt.xlabel('Flower Type')
plt.ylabel('Count')

plt.tight_layout()
plt.show()