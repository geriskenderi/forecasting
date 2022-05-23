import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


def plot_rolling_stats(ts, window):
    """
    Plot the rolling stats for a single time series.

    Parameters:
        ts: (TimeSeries) ts data as a pandas dataframe
        window: (int) temporal window for rolling statistic calculation (more or less = smoothing amount)
    Return:
        void
        Displays pyplot figure
    """

    plt.plot(ts, label='TS')
    plt.plot(ts.rolling(window=window).mean(), label="rolling mean")
    plt.plot(ts.rolling(window=window).std(), label="rolling std")
    plt.legend()
    plt.title('Rolling mean and std')
    plt.show()

def kpss_test(ts):
    """
    KPSS test to check for the stationarity of a time series

    Parameters:
        ts: (TimeSeries) ts data as a pandas dataframe

    Return:
        void
        Prints out test information
    """

    print(" > Is the data stationary ?")
    dftest = kpss(ts, 'ct', nlags='auto')
    print("Test statistic = {:.3f}".format(dftest[0]))
    print("P-value = {:.3f}".format(dftest[1]))
    print("Critical values :")
    for k, v in dftest[3].items():
        print("\t{}: {}".format(k, v))
    print("")


# Stampa dei grafici ACF e PACF
def autocorrelation(ts, nlags):
    """
    Plot both ACF and PACF plots for the given number of lags.

    Parameters:
        ts: (TimeSeries) ts data as a pandas dataframe
        lags: number of lags that both plots should contain

    Return:
        void
        Displays pyplot figure
    """

    fig, ax = plt.subplots(2, figsize=(12, 6))
    ax[0] = plot_acf(ts.dropna(), ax=ax[0], lags=nlags)
    ax[1] = plot_pacf(ts.dropna(), ax=ax[1], lags=nlags)
    plt.show()

def decompose_ts(ts):
    """
    Performs STL decomposition of the time series and plots the results

    Parameters:
        ts: (TimeSeries) contenente i dati da analizzare

    Return:
        void
        Displays pyplot figure        
    """

    decomposition = seasonal_decompose(ts)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(ts, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.title('Decomposed time-series')
    plt.show()
