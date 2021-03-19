import numpy as np

def pct_train_test_split(df, test_length=0.2):
    # Perform train-test split by keeping the specified percentage as test
    split_index = int(df.shape[0]*test_length)
    train = df[:-split_index]
    test = df[-split_index:]
    return train, test

def abs_train_test_keepnr(df, keepnr=7):
    # Perform train-test split by keeping the specified number as test
    train = df[:-keepnr]
    test = df[-keepnr:]
    return train, test

def smape(gt, pred):
    # Calculate Symmetrical Mean Absolute Percentage Error
    return 100 / len(gt) * np.sum(np.abs(pred - gt) / (np.abs(gt) + np.abs(pred)))