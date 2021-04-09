import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


class UnivariateTSDataset(object):
    def __init__(self, data, target_col, input_len, forecast_horizon):
        '''
        :param data: pandas.DataFrame containing the time series
        :param target_col: name of the targeted column
        :param input_len: window length to use
        :param forecast_horizon: window length to predict
        '''

        self.data = data
        self.target_col = target_col
        self.input_len = input_len
        self.forecast_horizon = forecast_horizon

    def preprocess_data(self):
        '''Preprocessing function'''
        ts = self.data[self.target_col].values
        scaler = MinMaxScaler()
        y = scaler.fit_transform(ts.reshape(-1, 1))
        train, test = train_test_split(y, shuffle=False)

        return train, test

    def frame_series(self, ts):
        '''
        Function used to prepare the data for time series prediction
        :param ts: univariate time series which we need to frame
        :return: TensorDataset
        '''

        nb_obs, nb_features = ts.shape
        inp, target, y_hist = [], [], []

        for i in range(1, nb_obs - self.input_len - self.forecast_horizon):
            inp.append(torch.FloatTensor(ts[i:i + self.input_len, :]).unsqueeze(0))

        inp_var = torch.cat(inp)

        for i in range(1, nb_obs - self.input_len - self.forecast_horizon):
            target.append(
                torch.FloatTensor(ts[i + self.input_len:i + self.input_len + self.forecast_horizon]).unsqueeze(0))

        target_var = torch.cat(target)

        return TensorDataset(inp_var, target_var)

    def get_loaders(self, batch_size: int):
        '''
        Preprocess and frame the dataset
        :param batch_size: batch size
        :return: DataLoaders associated to training and testing data
        '''
        train, test = self.preprocess_data()
        train_dataset = self.frame_series(train)
        test_dataset = self.frame_series(test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        return train_dataset, train_loader, test_dataset, test_loader


class MultivariateTSDataset(object):
    def __init__(self, data, target_col, continuous_cols, categorical_cols, input_len, forecast_horizon):
        '''
        :param data: pandas.DataFrame containing the time series
        :param target_col: name of the targeted column
        :param continuous_cols: list of the names of the continuous (numerical) columns
        :param categorical_cols: list of the names of the categorical columns, if None pass empty list
        :param input_len: window length to use
        :param forecast_horizon: window length to predict
        '''
        self.data = data
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols + [target_col]
        self.target_col = target_col
        self.input_len = input_len
        self.forecast_horizon = forecast_horizon
        self.preprocessor = None

    def preprocess_data(self):
        '''Preprocessing function'''
        X = self.data[self.continuous_cols]
        y = self.data[self.target_col]

        # Scale target data first
        y_scaler = MinMaxScaler()
        y = y_scaler.fit_transform(y.values.reshape(-1, 1))

        self.preprocessor = ColumnTransformer(
            [("scaler", MinMaxScaler(), self.continuous_cols),
             ("encoder", OneHotEncoder(), self.categorical_cols)],
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
        X_train, X_test, y_train, y_test = self.preprocess_data()

        train_dataset = self.frame_series(X_train, y_train)
        test_dataset = self.frame_series(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        return train_dataset, train_loader, test_dataset, test_loader
