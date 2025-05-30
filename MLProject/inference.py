from fastapi import FastAPI, Query, HTTPException
import joblib
import numpy as np
import os

app = FastAPI()

# Pastikan model tersedia
if not os.path.exists("model.pkl"):
    raise FileNotFoundError("❌ model.pkl tidak ditemukan!")

# Load model
model = joblib.load("model.pkl")

@app.get("/predict")
def predict(
    Pclass: int = Query(..., ge=1, le=3, description="Class of passenger (1, 2, or 3)"),
    Sex: int = Query(..., ge=0, le=1, description="0 = Female, 1 = Male"),
    Age: float = Query(..., ge=0, le=100, description="Age in years (0–100)"),
    SibSp: int = Query(..., ge=0, le=10, description="Number of siblings/spouses aboard"),
    Parch: int = Query(..., ge=0, le=10, description="Number of parents/children aboard"),
    Fare: float = Query(..., ge=0, description="Ticket fare (must be ≥ 0)"),
    Embarked: int = Query(..., ge=0, le=2, description="0 = S, 1 = C, 2 = Q (encoded)")
):
    try:
        features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
