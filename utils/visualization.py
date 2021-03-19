import matplotlib.pyplot as plt

def col_distributions(df):
    # Visualize all of the series in the dataframe
    for colname in df.columns:
        ts = df[colname].values
        plt.hist(ts, label=colname)
        plt.legend()
        plt.show()

    plt.close()