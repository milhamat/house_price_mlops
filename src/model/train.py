import os
import pickle
import mlflow 
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
from src.model.data_process import Preprocessing
from src.utils.config import Config
from src.utils.config import get_artifact_path
from src.utils.log import Logger
from src.utils.utils import log_system_metrics
from sklearn import ensemble

logger = Logger(Config.PROJECTNAME)

class TrainModel(): 
    def __init__(self, 
                 model_params:dict = {'n_estimators':3000, 
                                      'learning_rate':0.05, 
                                      'max_depth':3, 
                                      'max_features':'sqrt',
                                      'min_samples_leaf':15, 
                                      'min_samples_split':10, 
                                      'loss':'huber'},
                 dataset:str = Config.DATA_NAME, 
                 train_size:float = 0.1):
        self.model_params = model_params 
        self.dataset = dataset
        self.train_size = train_size
        
        
    def train_and_log_model(self):
        # Load dataset
        data_path = get_artifact_path(self.dataset)
        data = pd.read_csv(data_path)
        print(data.shape)
        # DATA PREPROCESSING
        try:
            logger.info("Starting data preprocessing...")
            train_features, train_labels = Preprocessing().preprocess(data)
            logger.success("Data preprocessing success...")
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise

        # Split the data into training and test sets 
        x_train, x_test, y_train, y_test = train_test_split(train_features, 
                                                            train_labels, 
                                                            test_size=self.train_size, 
                                                            random_state=0)

        params = self.model_params

        # Train the model
        logger.info("Starting model training...")
        gbr = ensemble.GradientBoostingRegressor(**params).fit(x_train, y_train)
        logger.success("Model training completed successfully!")
        # Predict on the test set
        y_pred = gbr.predict(x_test)

        # Evaluate the model
        logger.info("Evaluating the model...")
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        logger.info(f"R2 : {r2}")
        logger.info(f"RMSE : {rmse}")
        logger.info(f"MSE : {mse}")
        logger.info(f"MAE : {mae}")

        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        mlflow.set_experiment(Config.PROJECTNAME)
        mlflow.enable_system_metrics_logging()

        # Log model with MLflow
        with mlflow.start_run(nested=True):
            log_system_metrics(interval=5, max_iterations=5)
            mlflow.log_params(self.model_params)
            mlflow.log_metric("R2", r2)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("MAE", mae)
            
            mlflow.set_tag("Training Info", "house price prediction")
            # Infer and log model signature
            signature = infer_signature(x_train, gbr.predict(x_train))
            
            mlflow.sklearn.log_model(gbr, 
                                    artifact_path=Config.ARTIFACT_MODEL,
                                    signature=signature,
                                    input_example=x_test[:5],
                                    registered_model_name="tracking-quickstart-house-price-prediction",
                                    )
            
            # Log the training dataset as an artifact
            mlflow.log_artifact(data_path, artifact_path=Config.ARTIFACT_DATASET)

        # Save the model locally
        logger.info("model.pkl saving")
        model_path = get_artifact_path(Config.ARTIFACT_MODEL, Config.MODEL_NAME)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(gbr, f)
        logger.success("model.pkl saving")



