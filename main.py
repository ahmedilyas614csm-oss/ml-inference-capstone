from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# 1. Initialize the FastAPI app
app = FastAPI(
    title="ML Inference API",
    description="Real-Time ML Inference REST API for Iris Classification",
    version="1.0"
)

# 2. Load the trained model
try:
    model = joblib.load("model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# 3. Define the Input Schema
class PredictionInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# 4. Define the Output Schema
class PredictionOutput(BaseModel):
    prediction: int
    probability: float

# 5. Health check endpoint
@app.get("/")
def read_root():
    return {"status": "healthy"}

# 6. The main /predict endpoint
@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    try:
        input_data = np.array([[
            data.feature1,
            data.feature2,
            data.feature3,
            data.feature4
        ]])
        prediction = model.predict(input_data)[0]
        probability = float(model.predict_proba(input_data)[0].max())
        return PredictionOutput(
            prediction=int(prediction),
            probability=probability
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))