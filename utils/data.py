import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


def load_air_quality_data(path):
    df = pd.read_csv(path, sep=';')
    df = df.iloc[:, :-2]
    x = df.loc[:, ['Date', 'Time', 'T', 'RH', 'AH']]
    y = df.loc[:, 'PT08.S1(CO)']
    final_df = pd.concat((x, y), axis=1)
    final_df = final_df.dropna()  # drop nulls
    final_df = final_df[final_df['PT08.S1(CO)'] != -200]  # -200 is assigned to missing values

    # Build a datetime index
    dt_index = []
    for i in range(len(final_df)):
        date = final_df.iloc[i, 0]
        time = final_df.iloc[i, 1]
        date_time = date + ' ' + time
        dt_index.append(datetime.strptime(date_time, '%d/%m/%Y %H.%M.%S'))

    final_df['DT'] = dt_index
    final_df = final_df.set_index('DT')
    final_df = final_df.iloc[:, 2:]

    # Cast columns to correct type (numeric: float)
    for colname in ['T', 'RH', 'AH']:
        final_df[colname] = [str.replace(',', '.') for str in final_df[colname]]
    final_df = final_df.astype('float32')

    return final_df


def normalize_data(df):
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(df)
    return norm_data


def pct_train_test_split(df, test_length=0.2):
    # Perform train-test split by keeping the specified percentage as test
    split_index = int(df.shape[0] * test_length)
    train = df[:-split_index]
    test = df[-split_index:]
    return train, test


def abs_train_test_split(df, test_length=7):
    # Perform train-test split by keeping the specified number as test
    train = df[:-test_length]
    test = df[-test_length:]
    return train, test


def smape(gt, pred):
    # Calculate Symmetrical Mean Absolute Percentage Error
    return 100 / len(gt) * np.sum(np.abs(pred - gt) / (np.abs(gt) + np.abs(pred)))
