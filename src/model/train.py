import mlflow # type: ignore
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from src.utils.config import get_artifact_path
from src.utils.config import log_message
from sklearn import ensemble
import os
import pickle

def train_and_log_model():
    # Load dataset
    # data_path = os.path.join("artifacts", "train.csv")
    data_path = get_artifact_path("train.csv")
    train = pd.read_csv(data_path)

    # DATA PREPROCESSING
    selected = ['Id', 'LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt',
        'YearRemodAdd', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr',
        'TotRmsAbvGrd','SalePrice']
    train = train[selected]

    train_labels = train.pop('SalePrice')

    features = train
    features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].mean())
    train_labels = np.log(train_labels)
    train_features = features.drop('Id', axis=1).select_dtypes(include=[np.number]).values

    # Split the data into training and test sets 
    x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.1, random_state=0)

    params = {'n_estimators':3000, 'learning_rate':0.05, 'max_depth':3, 'max_features':'sqrt','min_samples_leaf':15, 'min_samples_split':10, 'loss':'huber'}

    # Train the model
    log_message("Starting model training...", "INFO")
    gbr = ensemble.GradientBoostingRegressor(**params).fit(x_train, y_train)
    log_message("Model training completed successfully!", "SUCCESS")
    # Predict on the test set
    y_pred = gbr.predict(x_test)

    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    mlflow.set_tag("Training Info", "Basic LR model for house price prediction")

    # Log model with MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.sklearn.log_model(gbr, 
                                 artifact_path="models")

        # print(f"Model logged with MSE: {mse}")

    # Save the model locally
    # model_path = os.path.join("artifacts", "models", "model.pkl")
    model_path = get_artifact_path("models", "model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(gbr, f)
