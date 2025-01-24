import joblib
from fastapi import FastAPI
import pandas as pd

app = FastAPI()

@app.post("/predict")
async def predict_rating(game_data: dict):
    model = joblib.load('game_rating_predictor.pkl')
    input_data = pd.DataFrame([game_data])
    prediction = model.predict(input_data)[0]
    return {"prediction": "High Rating" if prediction == 1 else "Low Rating"}