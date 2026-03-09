from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Create FastAPI instance
app = FastAPI()

# Define request body structure
class IrisFeatures(BaseModel):
    features: list

# Prediction endpoint
@app.post("/predict")
def predict(data: IrisFeatures):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    
    return {"prediction": int(prediction[0])}