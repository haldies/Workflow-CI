from fastapi import FastAPI, Query
import joblib
import numpy as np

app = FastAPI()

# Load model.pkl (pastikan file ada di container)
model = joblib.load("model.pkl")

@app.get("/predict")
def predict(
    Pclass: int = Query(...),
    Sex: int = Query(...),
    Age: float = Query(...),
    SibSp: int = Query(...),
    Parch: int = Query(...),
    Fare: float = Query(...),
    Embarked: int = Query(...)
):
    features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
