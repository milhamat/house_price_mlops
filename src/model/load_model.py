import os
import mlflow
import numpy as np
# from pathlib import Path


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
    

# load = LoadModel()

# model = load.load_model()

# model_path = os.path.abspath("artifacts/models")
# print(model_path)
# model = mlflow.pyfunc.load_model(model_path)
# print("Model loaded successfully from:", model_path)

# x_test = [6.93057953e+01, 3.26680000e+04, 6.00000000e+00, 1.95700000e+03,
#        1.97500000e+03, 2.51500000e+03, 3.00000000e+00, 0.00000000e+00,
#        4.00000000e+00, 9.00000000e+00]

# print(np.expm1(model.predict(np.array(x_test).reshape(1, 10))))
