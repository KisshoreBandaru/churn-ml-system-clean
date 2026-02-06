from fastapi import FastAPI
from app.model import ModelService
from app.schema import ChurnInput
# test change for git practice

app = FastAPI()

model_service = ModelService()  # model loads here ONCE

@app.post("/predict")
def predict(data: ChurnInput):
    features = [[
        data.tenure,
        data.monthly_charges,
        data.total_charges,
        data.support_calls
    ]]

    prediction = model_service.predict(features)
    return {"churn_prediction": prediction}
