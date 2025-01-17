from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Prediction request schema"""
    LotFrontage: float
    LotArea: float
    OverallQual: float
    YearBuilt: float
    YearRemodAdd: float
    GrLivArea: float
    FullBath: float
    HalfBath: float
    BedroomAbvGr: float
    TotRmsAbvGrd: float
    

class PredictionResponse(BaseModel):
    """Prediction response schema"""
    SalePrice: str
    prediction_timestamp: str
    prediction_id: str
    