import os
import mlflow


class LoadModel:
    def __init__(self, model_dir = "artifacts/models"):
        self.model_dir = model_dir

    def load_model(self):
        """
        Load the model from the specified directory using MLflow's pyfunc.
        """
        # Get the absolute path to the model
        model_path = os.path.abspath(self.model_dir)
        # model_path = self.model_dir
        try:
            # Load the model using MLflow
            model = mlflow.pyfunc.load_model(model_path)
            print("Model loaded successfully from:", model_path)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict(self, data):
        model = self.load_model()
        return model.predict(data)
