import torch
import torch.nn as nn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# --- 1. PANDAS: Data ---
data = {
    'age': [19, 18, 28, 33],
    'bmi': [27.9, 33.7, 33.0, 22.7],
    'smoker': [1, 0, 0, 1]
}
df = pd.DataFrame(data)
print("Data loaded:\n", df)

# --- 2. PYTORCH: The Brain ---
class InsuranceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

model = InsuranceModel()

# --- 3. PYTHON: Prediction Logic ---
def get_prediction(age, bmi, smoker):
    input_tensor = torch.tensor([[float(age), float(bmi), float(smoker)]])
    with torch.no_grad():
        prediction = model(input_tensor)
    return round(prediction.item(), 2)

# --- 4. FASTAPI: The Door ---
app = FastAPI()

class UserInput(BaseModel):
    age: int
    bmi: float
    smoker: int

@app.post("/predict")
def predict_cost(user: UserInput):
    cost = get_prediction(user.age, user.bmi, user.smoker)
    return {"status": "success", "estimated_insurance_premium": f"${cost}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)