import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def line_plot(x, show=False, savepath=None):
    '''
    Simple lineplot connecting all of the observations in time
    x: pandas DataFrame object containing the observations indexed by time
    '''
    sns.lineplot(x)
    
    if show:
        plt.show()
    if savepath:
        plt.savefig(savepath)
    plt.close()
    

def seasonal_plot(x, seasonality=365, show=False, savepath=None):
    if show:
        plt.show()
    if savepath:
        plt.savefig(savepath)
    plt.close()

def seasonal_subseries_plot(x, seasonality=365, show=False, savepath=None):
    if show:
        plt.show()
    if savepath:
        plt.savefig(savepath)
    plt.close()

def acf_plot(x, k=1, show=False, savepath=None):
    fig = plot_acf(x)

    if show:
        plt.show(fig)
    if savepath:
        plt.savefig(fig, savepath)
    plt.close()

def pacf_plot(x, k=1, show=False, savepath=None):
    fig = plot_pacf(x)

    if show:
        plt.show(fig)
    if savepath:
        plt.savefig(fig, savepath)
    plt.close()

def scatter_plot(x, y, show=False, savepath=None):
    '''
    Simple lineplot connecting all of the observations in time
    x: numpy sequence containing the observations of the first series
    y: numpy sequence containing the observations of the second series
    '''
    sns.scatterplot(x, y)

    if show:
        plt.show()
    if savepath:
        plt.savefig(savepath)
    plt.close()