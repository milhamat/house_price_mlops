import os
import pickle
import mlflow # type: ignore
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
from src.model.data_process import Preprocessing
from src.utils.config import get_artifact_path
from src.utils.config import log_message
from sklearn import ensemble

class TrainModel(): 
    def __init__(self, 
                 model_params:dict = {'n_estimators':3000, 
                                      'learning_rate':0.05, 
                                      'max_depth':3, 
                                      'max_features':'sqrt',
                                      'min_samples_leaf':15, 
                                      'min_samples_split':10, 
                                      'loss':'huber'},
                 dataset:str = "train.csv", 
                 train_size:float = 0.1):
        self.model_params = model_params 
        self.dataset = dataset
        self.train_size = train_size
        
        
    def train_and_log_model(self):
        # Load dataset
        data_path = get_artifact_path(self.dataset)
        data = pd.read_csv(data_path)

        # DATA PREPROCESSING
        try:
            log_message("Starting data preprocessing...", "INFO")
            train_features, train_labels = Preprocessing().preprocess(data)
            log_message("Data preprocessing success...", "SUCCESS")
        except Exception as e:
            log_message(f"Error in data preprocessing: {e}", "ERROR")
            raise

        # Split the data into training and test sets 
        x_train, x_test, y_train, y_test = train_test_split(train_features, 
                                                            train_labels, 
                                                            test_size=self.train_size, 
                                                            random_state=0)

        params = self.model_params

        # Train the model
        log_message("Starting model training...", "INFO")
        gbr = ensemble.GradientBoostingRegressor(**params).fit(x_train, y_train)
        log_message("Model training completed successfully!", "SUCCESS")
        # Predict on the test set
        y_pred = gbr.predict(x_test)

        # Evaluate the model
        log_message("Evaluating the model...", "INFO")
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        log_message(f"R2 : {r2}", "INFO")
        log_message(f"RMSE : {rmse}", "INFO")
        log_message(f"MSE : {mse}", "INFO")
        log_message(f"MAE : {mae}", "INFO")

        mlflow.set_tracking_uri('sqlite:///mlflow.db')

        # Log model with MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_metric("R2", r2)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("MAE", mae)
            
            mlflow.set_tag("Training Info", "house price prediction")
            # Infer and log model signature
            signature = infer_signature(x_train, gbr.predict(x_train))
            
            mlflow.sklearn.log_model(gbr, 
                                    artifact_path="models",
                                    signature=signature,
                                    input_example=x_test[:5])
            
            # Log the training dataset as an artifact
            train_data_path = get_artifact_path("train.csv")
            mlflow.log_artifact(train_data_path, artifact_path="datasets")

        # Save the model locally
        log_message("model.pkl saving", "INFO")
        model_path = get_artifact_path("models", "model.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(gbr, f)
        log_message("model.pkl saving", "SUCCESS")



