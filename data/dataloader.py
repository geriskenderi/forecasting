import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

class TSDataset(object):
    def __init__(self, data, continuous_cols_idx, categorical_cols_idx, 
    target_cols_idx, input_len, forecast_horizon):
        '''
        :param data: pandas.DataFrame containing the time series
        :param continuous_cols_idx: list of the indices of the continuous columns
        :param categorical_cols_idx: list of the indices of the categorical columns (if None == empty list)
        :param target_cols_idx: list of the indices of the target columns (variables to be forecasted)
        :param input_len: length of input time-series
        :param forecast_horizon: length of forecasted series

        '''
        self.data = data
        self.continuous_cols_idx = continuous_cols_idx
        self.categorical_cols_idx = categorical_cols_idx
        self.target_cols_idx = target_cols_idx
        self.input_len = input_len
        self.forecast_horizon = forecast_horizon
        X_train, X_test, y_train, y_test = self.preprocess_data()
        train_dataset, test_dataset = self.frame_series(X_train, y_train), self.frame_series(X_test, y_test)

    def preprocess_data(self):
        '''Preprocessing function'''
        input_cols = list(set(self.continuous_cols_idx + self.categorical_cols_idx + self.target_cols_idx))
        X = self.data.iloc[:, input_cols]
        y = self.data.iloc[:, self.target_cols_idx]

        # Scale target data first
        ts_scaler = MinMaxScaler().fit(X.iloc[:, self.target_cols_idx])
        X.iloc[:, self.target_cols_idx] = ts_scaler.transform(X.iloc[:, self.target_cols_idx])
        X.iloc[:, self.continuous_cols_idx] = MinMaxScaler().fit_transform(X.iloc[:, self.continuous_cols_idx])
        X.iloc[:, self.categorical_cols_idx] = OneHotEncoder().fit_transform(X.iloc[:, self.categorical_cols_idx])
        y = ts_scaler.transform(y)

        self.preprocessor = ColumnTransformer(
            [("scaler", MinMaxScaler(), self.continuous_cols_idx),
             ("encoder", OneHotEncoder(), self.categorical_cols_idx)],
            remainder="passthrough"
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.fit_transform(X_test)

        if self.target_col:
            return X_train, X_test, y_train, y_test
        return X_train, X_test

    def frame_series(self, X, y=None):
        '''
        Function used to prepare the data for time series prediction
        :param X: set of features
        :param y: targeted value to predict
        :return: TensorDataset
        '''
        nb_obs, nb_features = X.shape
        features, target, y_hist = [], [], []

        for i in range(1, nb_obs - self.input_len - self.forecast_horizon):
            features.append(torch.FloatTensor(X[i:i + self.input_len, :]).unsqueeze(0))

        features_var = torch.cat(features)

        if y is not None:
            for i in range(1, nb_obs - self.input_len - self.forecast_horizon):
                target.append(
                    torch.FloatTensor(y[i + self.input_len:i + self.input_len + self.forecast_horizon]).unsqueeze(0))
            target_var = torch.cat(target)
            return TensorDataset(features_var, target_var)

        return TensorDataset(features_var)

    def get_loaders(self, batch_size: int):
        '''
        Preprocess and frame the dataset
        :param batch_size: batch size
        :return: DataLoaders associated to training and testing data
        '''
        
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        return train_loader, test_loader
