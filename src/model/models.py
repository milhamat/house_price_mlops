import os
import pickle
import pandas as pd
from src.utils.log import Logger

logger = Logger("House Price Prediction v.1")
class LoadModel:
    def __init__(self, model_path:str = "artifacts"):
        self.model_path = model_path

    def load_model(self):
        """
        Load the model from the specified directory using MLflow's pyfunc.
        """
        # Get the absolute path to the model
        try:
            # Load the model using MLflow
            logger.info("Model loaded...", "INFO")
            model_path = os.path.join(f"{self.model_path}", "models", "model.pkl")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.success(f"Model loaded successfully from: {self.model_path}", "SUCCESS")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            logger.error(f"Error loading model: {e}", "ERROR")
            raise
    
    def predict(self, data:pd.DataFrame)-> pd.DataFrame:
        logger.info("Model predicting..", "INFO")
        model = self.load_model()
        logger.success("Model predicting successfully...", "SUCCESS")
        return model.predict(data)
