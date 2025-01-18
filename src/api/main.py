import uuid
import numpy as np
import pandas as pd
from src.model.models import LoadModel
from fastapi import FastAPI, HTTPException
from src.api.schema import PredictionRequest, PredictionResponse

app = FastAPI(title="house price prediction API")

model = LoadModel()

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionRequest)-> PredictionResponse:
    """Make a prediction using the trained model.

    Args:
        data (PredictionRequest): The input data for making the prediction.

    Raises:
        HTTPException: An error occurred while making the prediction.

    Returns:
        _type_: The prediction response.
    """
    try:
        # Convert the input data to a pandas DataFrame
        input_data = data.dict()
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)
        prediction = np.expm1(prediction) 
        
        # Create response
        response = PredictionResponse(
            SalePrice=f'${int(prediction[0])}',
            prediction_timestamp=str(pd.Timestamp.now()),
            prediction_id=str(uuid.uuid4())
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return response


