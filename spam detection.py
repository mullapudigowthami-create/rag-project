# Email Spam Detection using AI
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Sample Email Data
emails = [
    "Win free money now click here",
    "Congratulations you won lottery",
    "Free prize claim immediately",
    "Buy cheap medicines online",
    "Make money fast easy way",
    "Meeting tomorrow at 10am",
    "Please review the attached report",
    "Team lunch is scheduled today",
    "Your invoice is attached",
    "Project deadline is next week",
    "Free iPhone winner selected",
    "Urgent bank account suspended",
    "Hello how are you doing",
    "Can we meet tomorrow morning",
    "Please complete your timesheet"
]

labels = [1,1,1,1,1,  # 1 = Spam
          0,0,0,0,0,  # 0 = Not Spam
          1,1,0,0,0]

# Create DataFrame
df = pd.DataFrame({
    'Email': emails,
    'Label': labels
})

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Email'])
y = df['Label']

# Split Data
X_train,X_test,y_train,y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Test new emails
new_emails = [
    "Win free cash prize now",
    "Meeting is scheduled for Monday"
]
new_X = vectorizer.transform(new_emails)
predictions = model.predict(new_X)

for email, pred in zip(new_emails, predictions):
    result = "SPAM" if pred == 1 else "NOT SPAM"
    print(f"\nEmail: {email}")
    print(f"Result: {result}")

# Plot spam vs not spam
plt.figure(figsize=(8,6))
counts = df['Label'].value_counts()
plt.bar(['Not Spam','Spam'],
        counts.values,
        color=['green','red'])
plt.title('Spam vs Not Spam Emails')
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.show()