import os
import pickle
import pandas as pd
from src.utils.config import Config

config = Config()

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
            config.log_message("Model loaded...", "INFO")
            model_path = os.path.join(f"{self.model_path}", "models", "model.pkl")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            config.log_message(f"Model loaded successfully from: {self.model_path}", "SUCCESS")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            config.log_message(f"Error loading model: {e}", "ERROR")
            raise
    
    def predict(self, data:pd.DataFrame)-> pd.DataFrame:
        config.log_message("Model predicting..", "INFO")
        model = self.load_model()
        config.log_message("Model predicting successfully...", "SUCCESS")
        return model.predict(data)
