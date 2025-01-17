import os
import pickle
from src.utils.config import log_message


class LoadModel:
    def __init__(self, model_path = "artifacts"):
        self.model_path = model_path

    def load_model(self):
        """
        Load the model from the specified directory using MLflow's pyfunc.
        """
        # Get the absolute path to the model
        try:
            # Load the model using MLflow
            model_path = os.path.join(f"{self.model_path}", "models", "model.pkl")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            log_message(f"Model loaded successfully from: {self.model_path}", "INFO")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            log_message(f"Error loading model: {e}", "INFO")
            raise
    
    def predict(self, data):
        model = self.load_model()
        return model.predict(data)
