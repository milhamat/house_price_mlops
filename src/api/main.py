import os 
import sys
import uuid
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from model.models import LoadModel
from fastapi import FastAPI, HTTPException
from schema import PredictionRequest, PredictionResponse

app = FastAPI(title="house price prediction API")

model = LoadModel()

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionRequest):
    """
    Predict the house price using the input data.
    """
    try:
        # Convert the input data to a pandas DataFrame
        input_data = data.dict()
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Create response
        response = PredictionResponse(
            SalePrice=prediction,
            prediction_timestamp=str(pd.Timestamp.now()),
            prediction_id=str(uuid.uuid4())
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return response



