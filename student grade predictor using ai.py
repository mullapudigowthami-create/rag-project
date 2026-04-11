 # ============================================
# Project 3: Student Grade Predictor using AI
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ============================================
# Step 1: Create Sample Data
# ============================================
# Features: [study_hours, attendance_percentage, sleep_hours]
X = np.array([
    [2, 60, 5],
    [3, 70, 6],
    [4, 75, 6],
    [5, 80, 7],
    [6, 85, 7],
    [7, 88, 8],
    [8, 90, 8],
    [9, 92, 8],
    [1, 50, 4],
    [3, 65, 6],
    [5, 78, 7],
    [6, 82, 7],
    [7, 87, 8],
    [4, 72, 6],
    [8, 95, 9],
])

# Target: Final Grade (out of 100)
y = np.array([42, 55, 62, 70, 75, 80, 85, 90, 30, 52, 68, 74, 82, 60, 95])

# ============================================
# Step 2: Split Data into Train & Test Sets
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=" * 40)
print("   Student Grade Predictor using AI")
print("=" * 40)
print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ============================================
# Step 3: Train the Model
# ============================================
model = LinearRegression()
model.fit(X_train, y_train)

print("\n✅ Model trained successfully!")

# ============================================
# Step 4: Evaluate the Model
# ============================================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("\n--- Model Accuracy ---")
print(f"Mean Absolute Error : {mae:.2f}")
print(f"R² Score            : {r2:.2f}  (1.0 = perfect)")

# ============================================
# Step 5: Predict New Students
# ============================================
print("\n--- Predicting New Students ---")
new_students = np.array([
    [5, 80, 7],   # Student A
    [2, 55, 5],   # Student B
    [9, 95, 8],   # Student C
])

names = ["Student A", "Student B", "Student C"]
predictions = model.predict(new_students)

for name, hours, pred in zip(names, new_students, predictions):
    grade = round(pred, 1)
    status = "✅ PASS" if grade >= 50 else "❌ FAIL"
    print(f"{name} | Study: {hours[0]}h | Attendance: {hours[1]}% | "
          f"Sleep: {hours[2]}h → Predicted Grade: {grade} {status}")

# ============================================
# Step 6: Visualize Results
# ============================================

# --- Chart 1: Actual vs Predicted (Test Set) ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', edgecolors='black', s=100)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")
plt.title("Actual vs Predicted Grades")
plt.legend()
plt.grid(True)

# --- Chart 2: Study Hours vs Grade ---
plt.subplot(1, 2, 2)
study_range = np.linspace(1, 10, 100).reshape(-1, 1)
# Fix attendance & sleep at average for visualization
avg_attendance = np.full((100, 1), 78)
avg_sleep      = np.full((100, 1), 7)
X_vis = np.hstack([study_range, avg_attendance, avg_sleep])
grade_line = model.predict(X_vis)

plt.scatter(X[:, 0], y, color='blue', label='Actual Data', zorder=5)
plt.plot(study_range, grade_line, color='red',
         linestyle='--', label='AI Prediction Line')
plt.xlabel("Study Hours per Day")
plt.ylabel("Grade (out of 100)")
plt.title("Study Hours vs Grade")
plt.axhline(y=50, color='green', linestyle=':', linewidth=1.5, label='Pass Mark (50)')
plt.legend()
plt.grid(True)

plt.suptitle("Student Grade Predictor using AI", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show() 