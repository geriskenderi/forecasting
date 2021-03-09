from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime  # gestione date
import matplotlib.pyplot as plt  # stampa grafici
from statsmodels.tsa.stattools import adfuller
import pandas as pd  # gestione time series
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SARIMAX
from pandas.plotting._matplotlib import autocorrelation_plot
from math import sqrt
from statsmodels.tsa.holtwinters import Holt
import pmdarima as pm
import time
from time import gmtime, strftime
from numpy import mean


#Implementazione Naive Method
def naive_method(train, test, article='NAIVE METHOD FORECASTING'):
    """Funzione che implementa il metodo Naive (previsione basata sull'ultimo valore del trainset).
    Stampa a video il grafico di trainset, testset e previsione.

    Parametri:
        train: (TimeSeries) contenente il trainset
        test: (TimeSeries) contenente il testset
        article: (str) titolo del grafico stampato

    Return: 
        df_forecast: (DataFrame) contenente i valori della previsione e relativo errore
    """
    naive_val = []
    for i in range(len(test)):
        naive_val.append(train[-1])
    ts_naive = pd.Series(naive_val, index=test.index)

    df_forecast = pd.DataFrame.from_dict({'Values': test.values, 'Forecast':naive_val, 'Error': test.values - naive_val})
    df_forecast.index = test.index

    plt.plot(train, label='TRAIN VAL.')
    plt.plot(test, label='TEST VAL.')
    plt.plot(ts_naive, label='NAIVE')
    plt.title(article)
    plt.legend()
    plt.show()

    return df_forecast

#Stampa dei grafici di rolling mea e rolling std
def plot_rolling_stats(ts, article='ROLLING STATS'):
    """Funzione che stampa a video le rolling stats della serie specificata.

    Parametri:
        ts: (TimeSeries) contenente i dati da analizzare
        article: (str) titolo del grafico stampato

    Return:
        -
    """

    plt.plot(ts, label='TS')
    plt.plot(ts.rolling(window=52).mean(), label="rolling mean")
    plt.plot(ts.rolling(window=52).std(), label = "rolling std")
    plt.legend()
    plt.title(article)
    plt.show()

#Implementazione Dickey-Fuller test (verifica della 'stazionarietà' di una serie)
def df_test_stationarity(timeseries, article='DF test'):
    """Funzione che stampa a video il risultato del DF test eseguito sulla serie indicata.

    Parametri:
        timeseries: (TimeSeries) contenente i dati da analizzare
        article: (str) titolo del resoconto stampato

    Return:
        -
    """
    #Dickey-Fuller test:
    print('\n' + article)
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    print('---------------- Results ----------------')
    print('The data is {}stationary with 99% confidence'.format('NOT ' if dftest[4]['1%'] < dftest[0] else ''))
    print('The data is stationary with 95% confidence'.format('NOT ' if dftest[4]['5%'] < dftest[0] else ''))
    print('The data is stationary with 90% confidence'.format('NOT ' if dftest[4]['10%'] < dftest[0] else ''))
    print('\n')

#Implementazione KPSS test (verifica della 'stazionarietà' di una serie)
def kpss_test(ts, article='KPSS test'):
    """Funzione che stampa a video il risultato del KPSS test eseguito sulla serie indicata.

    Parametri:
        ts: (TimeSeries) contenente i dati da analizzare
        article: (str) titolo del resoconto stampato

    Return:
        -
    """
    print("\n", article)
    print(" > Is the data stationary ?")
    dftest = kpss(ts, 'ct', nlags='auto')
    print("Test statistic = {:.3f}".format(dftest[0]))
    print("P-value = {:.3f}".format(dftest[1]))
    print("Critical values :")
    for k, v in dftest[3].items():
        print("\t{}: {}".format(k, v))
    print("")

#Implementazione del metodo Simple Exponential Smoothing
def simple_exp_smooth(train, test, alpha=0.4, extra_periods=32, article='SimpleExonentialSmooth'):
    """Funzione che implementa il metodo Simple Exponential Smoothing. La previsione è una media pesata dei valori precedenti,
    regolata dal valore di alpha (compreso tra 0 e 1, indica quanto peso viene dati ai valori recenti rispetto a quelli più vecchi).

    Parametri:
        train: (TimeSeries) contenente i dati del trainset
        test: (TimeSeries) contenente i dati del testset
        alpha: (float, 0 < alpha < 1) parametro
        extra_periods: (int) numero di valori da prevedere
        article: (str) titolo del grafico stampato

    Return:
        df_forecast: (DataFrame) contenente i valori della previsione e relativi errori
    """
    d = np.array(train)  # Transform the input into a numpy array
    cols = len(d) # Historical period length
    # Append np.nan into the demand array to cover future periods
    d = np.append(d, [np.nan]*extra_periods)
    f = np.full(cols+extra_periods, np.nan)  # Forecast array
    f[1] = d[0]  # initialization of first forecast
    # Create all the t+1 forecasts until end of historical period
    for t in range(2, cols+1):
        f[t] = alpha*d[t-1]+(1-alpha)*f[t-1]
    f[cols+1:] = f[t]  # Forecast for all extra periods

    df_forecast = pd.DataFrame.from_dict({"Values": train.append(test), "Forecast": f, "Error": train.append(test)-f})
    df_forecast.index = df.index

    plt.plot(train, label='TRAIN VAL.')
    plt.plot(test, label='TEST VAL.')
    plt.plot(df_forecast['Forecast'], label='S.E.M.')
    plt.legend()
    plt.title(article)
    plt.show()

    return df_forecast

#Implementazione del metodo Holt's Linear Smoothing
def holt(train, test, article='HOLT'):
    model = Holt(train.values)
    model._index = pd.to_datetime(train.index)

    fit1 = model.fit(smoothing_level=.3, smoothing_trend=.05)
    pred1 = fit1.forecast(9)
    fit2 = model.fit(optimized=True)
    pred2 = fit2.forecast(9)
    fit3 = model.fit(smoothing_level=.3, smoothing_trend=.2)
    pred3 = fit3.forecast(9)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train.values)
    ax.plot(test.index, test.values, color="gray")
    for p, f, c in zip((pred1, pred2, pred3), (fit1, fit2, fit3), ('#ff7823', '#3c763d', 'c')):
        ax.plot(train.index[196:], f.fittedvalues[196:], color=c)
        ax.plot(test.index, p, label="alpha="+str(f.params['smoothing_level'])[:4]+", beta="+str(f.params['smoothing_trend'])[:4], color=c)
    plt.title(article)
    plt.legend()

#Stampa dei grafici ACF e PACF
def autocorrelation(ts):
    """Funzione che stampa a video i grafici ACF e PACF.

    Parametri:
        ts: (TimeSeries) contenente i dati da analizzare

    Return:
        -
    """
    fig, ax = plt.subplots(2, figsize=(12, 6))
    ax[0] = plot_acf(ts.dropna(), ax=ax[0], lags=20)
    ax[1] = plot_pacf(ts.dropna(), ax=ax[1], lags=20)
    plt.show()

#Stampa dei grafici ACF e PACF
#def plot_acf_pacf(ts, article = ''):
    lag_acf = acf(ts, nlags=20, fft=False)
    lag_pacf = pacf(ts, nlags=20, method='ols')

    #Plot ACF:
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.title('Autocorrelation Function ' + article)

    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function ' + article)
    plt.tight_layout()
    plt.show()

#Stampa la scomposizione della serie in trend, stagionalità, residuals
def decomposing_ts(ts, article=''):
    """Funzione che stampa a video la scomposizione della serie in trend, stagionalità, residuals.

    Parametri:
        ts: (TimeSeries) contenente i dati da analizzare
        article: (str) titolo del grafico stampato

    Return:
        -
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
    plt.title(article)
    plt.show()

def arima(p, d, q, train, test):
    """Funzione che esegue il metodo ARIMA sulla serie specificata, stampando il grafico della previsione
    e statistiche sulla precisione della previsione.

    Parametri:
        p, d, q: (int) parametri per eseguire ARIMA
        train: (TimeSeries) contenente i dati del trainset
        test: (TimeSeries) contenente i dati del testset

    Return:
        error: (DataFrame) contenente la previsione e i relativi errori
    """
    model = ARIMA(train, order=(p,d,q))
    fmodel = model.fit()
    print(fmodel.summary())
    pred = pd.Series(fmodel.predict('2019-05-13', '2020-05-11'))
    error = pd.DataFrame.from_dict({'Values': test.values, 'Forecast': pred.values, 'Error': test.values - pred.values})
    error.index = test.index

    mae = np.mean(abs(error['Error']))
    rmse = np.sqrt(np.mean(error['Error']**2))
    nrmse = rmse / (max(test) - min(test))

    print("RMSE:", rmse, "\nMAE:", mae, "\nNRMSE:", nrmse)

    plt.plot(pred, label='prediction')
    plt.plot(test, label='expected values')
    plt.plot(train, label='train set')
    plt.legend()
    plt.show()

    return error

#AUTO-ARIMA
def auto_arima(df, train):
    """Funzione che esegue il metodo ARIMA sul DataFrame delle vendite di articoli, stampando il grafico della previsione
    e statistiche sulla precisione della previsione.

    Parametri:
        df: (DataFrame) contenente i dati da analizzare
        train: (int) dimensione trainset

    Return:
        error: (DataFrame) contenente la previsione e i relativi errori
    """
    nomi_art = ['MAGLIE', 'GIACCHE', 'PANTALONI','GONNE', 'VESTITI', 'CAMICIE']

    tot_mae = []
    tot_rmse = []
    tot_nrmse = []

    for art in nomi_art:

        print("\nFORECASTING",art,"...")

        ts = to_ts_article(art, df)
        trainset = ts[:train]
        testset = ts[train:]

        model = pm.auto_arima(trainset, seasonal=True, m=52)
        print('\n+++++++++++++++++++++++++++++++++++++++',art,"+++++++++++++++++++++++++++++++++++++++++++++++++\n")
        print(model.summary())
        forecast = model.predict(len(testset))

        error = testset.values - forecast

        mae = np.mean(abs(error))
        rmse = np.sqrt(np.mean(error**2))
        nrmse = rmse / (max(testset) - min(testset))

        tot_mae.append(mae)
        tot_rmse.append(rmse)
        tot_nrmse.append(nrmse)

        #plt.plot(trainset, label='Train set')
        #plt.plot(testset, label='Test set')
        #plt.plot(pd.Series(forecast, index=testset.index), label='Predictions')
        #plt.legend()
        #plt.title(art)
        #plt.show()

    return pd.DataFrame.from_dict({'ARTICOLO': nomi_art, 'RMSE': tot_rmse, 'Normalized-RMSE':tot_nrmse, 'MAE': tot_mae})

#print("\n", auto_arima_article(df_art_short, 157), "\n")

#   Funzione che prende in input il dataframe "per capo" e esegue ARIMA selezionando in automatico i parametri più efficaci.
#   Ritorna un dataframe con le statistiche sulle previsioni.
def auto_arima_capo(df, train=18):
    """Funzione che esegue il metodo ARIMA sul DataFrame delle vendite di capi, stampando il grafico della previsione
    e statistiche sulla precisione della previsione.

    Parametri:
        df: (DataFrame) contenente i dati da analizzare
        train: (int) dimensione trainset

    Return:
        error: (DataFrame) contenente la previsione e i relativi errori
    """
    tot_mae = []
    tot_rmse = []

    for line in df.index:

        print(line,'of',len(df.index) , '- FORECASTING', df['codice esterno'][line], "...")

        val = []
        for v in range (1,21):
            val.append(df[df.columns[v]][line])

        ts = to_ts_capo(val, df['First_week'][line])
        trainset = ts[:train]
        testset = ts[train:]

        model = pm.auto_arima(trainset)
        #print(model.summary())
        forecast = model.predict(len(testset))

        error = testset.values - forecast

        mae = np.mean(abs(error))
        rmse = np.sqrt(np.mean(error**2))

        tot_mae.append(round(mae, 2))
        tot_rmse.append(round(rmse, 2))
        #r2 = 1-(sum(error**2)/sum((testset.values-np.mean(testset.values))**2))

        #plt.plot(trainset, label='Train set')
        #plt.plot(testset, label='Test set')
        #plt.plot(pd.Series(forecast, index=testset.index), label='Predictions')
        #plt.legend()
        #plt.title(art)
        #plt.show()

    return pd.DataFrame.from_dict({'ARTICOLO': df['codice esterno'], 'RMSE': tot_rmse, 'MAE': tot_mae})

#print("\n", auto_arima_capo(df_capo), "\n")

