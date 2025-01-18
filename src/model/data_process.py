import numpy as np

class Preprocessing:
    def __init__(self, selected):
        self.selected = selected

    def preprocess(self, data):
        
        self.selected = ['Id', 'LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt',
        'YearRemodAdd', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr',
        'TotRmsAbvGrd','SalePrice']
        train = data[self.selected]

        train_labels = train.pop('SalePrice')

        features = train
        features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].mean())
        train_labels = np.log(train_labels)
        train_features = features.drop('Id', axis=1).select_dtypes(include=[np.number]).values
        
        return train_features, train_labels