import pandas as pd
import numpy as np
import pmdarima as pm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import Holt


def naive(train, test):
    """
    Function that implements the Naive method (forecast based on the last trainset value).
    Plots the train, test and forecast series on the screen.

    Parameters:
        train: (TimeSeries) train data
        test: (TimeSeries) test data

    Return:
        forecast: (DataFrame) containing the ground truth values and the forecast
    """
    naive_val = []
    for i in range(len(test)):
        naive_val.append(train[-1])
    ts_naive = pd.Series(naive_val, index=test.index)

    forecast = pd.DataFrame.from_dict({'GT': test.values, 'Forecast': naive_val})
    forecast.index = test.index

    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(ts_naive, label='Forecast')
    plt.title("Naive forecast")
    plt.legend()
    plt.show()

    return forecast


def holt(train, test):
    """
    Function that implements the Holt-Winter's exponential smoothing method.
    Plots the train, test and forecast series on the screen.

    Parameters:
        train: (TimeSeries) train data
        test: (TimeSeries) test data

    Return:
        forecast: (DataFrame) containing the ground truth values and the forecast
    """
    f_horizon = len(test)
    model = Holt(train.values)
    model._index = pd.to_datetime(train.index)

    fit = model.fit(optimized=True)
    pred = fit.forecast(f_horizon)

    forecast = pd.DataFrame.from_dict({'GT': test.values, 'Forecast': pred})
    forecast.index = test.index

    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(pred, label='Forecast')
    plt.title("Holt-Winters Smoothing")
    plt.legend()
    plt.show()

    return forecast


def arima(p, d, q, train, test):
    """
    Function that performs the ARIMA method on the given series.
    Plots the train, test and forecast series on the screen.

    Parameters:
        p, d, q: (int) ARIMA parameters
        train: (TimeSeries) train data
        test: (TimeSeries) test data

    return:
        forecast: (DataFrame) containing the ground truth values and the forecast
    """
    model = ARIMA(train, order=(p, d, q))
    fmodel = model.fit()
    print(fmodel.summary())

    pred = pd.Series(fmodel.predict(test))
    forecast = pd.DataFrame.from_dict({'GT': test.values, 'Forecast': pred.values})
    forecast.index = test.index

    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(pred, label='Forecast')
    plt.title("ARIMA ({}, {}, {})".format(p, d, q))
    plt.legend()
    plt.show()

    return forecast


def auto_arima(train, test):
    """
    Function that performs the Auto-ARIMA method on the given series. It automatically finds the best p,d and q for
    the series in a grid-search manner. More expensive than the standard arima function.
    Plots the train, test and forecast series on the screen.

    Parameters:
        p, d, q: (int) ARIMA parameters
        train: (TimeSeries) train data
        test: (TimeSeries) test data

    return:
        forecast: (DataFrame) containing the ground truth values and the forecast
    """
    f_horizon = len(test)
    model = pm.auto_arima(train)
    arima_order = model.get_params()["order"]

    pred = pd.Series(model.predict(f_horizon))
    forecast = pd.DataFrame.from_dict({'GT': test.values, 'Forecast': pred.values})
    forecast.index = test.index

    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(pred, label='Forecast')
    plt.title("Auto-ARIMA" + str(arima_order))
    plt.legend()
    plt.show()

    return forecast
