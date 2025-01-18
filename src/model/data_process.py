import numpy as np
import pandas as pd

class Preprocessing:
    def __init__(self):
        pass

    def preprocess(self, data:pd.DataFrame)-> tuple:
        
        selected = ['Id', 'LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt',
        'YearRemodAdd', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr',
        'TotRmsAbvGrd','SalePrice']
        train = data[selected]

        train_labels = train.pop('SalePrice')

        features = train
        features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].mean())
        train_labels = np.log(train_labels)
        train_features = features.drop('Id', axis=1).select_dtypes(include=[np.number]).values
        
        return train_features, train_labels