# Adding needed libraries and reading data
import mlflow
import numpy as np
import pandas as pd
from sklearn import ensemble
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

train = pd.read_csv('./artifacts/train.csv')

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
gbr = ensemble.GradientBoostingRegressor(**params).fit(x_train, y_train)

# Predict on the test set
y_pred = gbr.predict(x_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Set our model artifact folder
artifact_folder = "./artifacts/models"

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("MAE", mae)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for house price prediction")

    # Infer the model signature
    signature = infer_signature(x_train, gbr.predict(x_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=gbr,
        artifact_path="house_pred",
        signature=signature,
        input_example=x_train,
        registered_model_name="tracking-quickstart-house-price-prediction",
    )
    
    mlflow.sklearn.save_model(sk_model=gbr, path=artifact_folder)