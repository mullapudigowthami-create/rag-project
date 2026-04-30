import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# ======================================================
# 1. PANDAS: DATA PREPARATION (The Foundation)
# ======================================================
# In a real scenario, you'd use: df = pd.read_csv('insurance.csv')
# Here we create a dataset to show how Pandas handles features
data = {
    'age': [19, 18, 28, 33, 32, 31, 46, 37],
    'bmi': [27.9, 33.7, 33.0, 22.7, 28.8, 25.7, 33.4, 27.7],
    'smoker': [1, 0, 0, 0, 0, 0, 1, 0],
    'charges': [16884.9, 1725.5, 4449.4, 21984.4, 3866.8, 3756.6, 29141.3, 7281.5]
}
df = pd.DataFrame(data)

# Convert Pandas columns to NumPy arrays, then to PyTorch Tensors
X = df[['age', 'bmi', 'smoker']].values.astype(np.float32)
y = df[['charges']].values.astype(np.float32)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# ======================================================
# 2. PYTORCH: THE AI MODEL (The Brain)
# ======================================================
class InsuranceNet(nn.Module):
    def __init__(self):
        super(InsuranceNet, self).__init__()
        # 3 inputs (age, bmi, smoker) -> 10 hidden neurons -> 1 output (price)
        self.layer1 = nn.Linear(3, 10)
        self.layer2 = nn.Linear(10, 1)
        self.relu = nn.ReLU() # Activation function (Theory: adds non-linearity)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)

# Initialize Model, Loss Function, and Optimizer
model = InsuranceNet()
criterion = nn.MSELoss()  # Mean Squared Error (Theory: measures prediction error)
optimizer = optim.Adam(model.parameters(), lr=0.1) # Optimizer (Logic: updates weights)

# THE TRAINING LOOP
print("--- Starting AI Training ---")
for epoch in range(500):
    optimizer.zero_grad()           # Reset gradients
    predictions = model(X_tensor)   # Forward pass (Predict)
    loss = criterion(predictions, y_tensor) # Calculate Loss
    loss.backward()                 # Backward pass (Calculate gradients)
    optimizer.step()                # Update weights
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/500], Loss: {loss.item():.4f}")
print("--- Training Complete ---")

# ======================================================
# 3. FASTAPI: THE DEPLOYMENT (The Door)
# ======================================================
app = FastAPI()

# Pydantic model for data validation (Python Logic)
class PatientData(BaseModel):
    age: int
    bmi: float
    smoker: int # 1 for Yes, 0 for No

@app.get("/")
def home():
    return {"message": "AI Insurance Predictor is Live. Use /docs to test."}

@app.post("/predict")
def predict_cost(patient: PatientData):
    # Prepare the input for the model
    input_features = torch.tensor([[float(patient.age), patient.bmi, float(patient.smoker)]])
    
    # Set model to evaluation mode and predict
    model.eval()
    with torch.no_grad():
        prediction = model(input_features)
    
    return {
        "input_received": patient,
        "estimated_premium": 
        f"${round(prediction.item(), 2)}"
    }

# ======================================================
# HOW TO RUN THIS
# ======================================================
# 1. Install requirements: pip install torch pandas fastapi uvicorn
# 2. Save this code as: main.py
# 3. In your terminal, run: uvicorn main:app --reload
# 4. Open your browser at: http://127.0.0
