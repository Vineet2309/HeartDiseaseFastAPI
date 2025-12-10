#pip install fastapi uvicorn joblib pydantic
#python -m uvicorn MLFastAPI:app --reload --host 0.0.0.0 --port 8000
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
app = FastAPI()

# Load model and scaler
model = joblib.load("heart_modle.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("chestPainLE.pkl")

# Input schema
class HeartData(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

@app.post("/predict")
def predict(data: HeartData):

    df = pd.DataFrame([data.dict()])

    df['Sex'] = df['Sex'].map({'M':1,'F':0})
    df['ChestPainType'] = le.transform(df['ChestPainType'])
    df['RestingECG'] = df['RestingECG'].map({'Normal':0,'ST':1,'LVH':2})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y':1,'N':0})
    df['ST_Slope'] = df['ST_Slope'].map({'Up':0,'Flat':1,'Down':2})

    df_scaled = scaler.transform(df)
    prediction = int(model.predict(df_scaled)[0])

    return {"prediction": prediction}
