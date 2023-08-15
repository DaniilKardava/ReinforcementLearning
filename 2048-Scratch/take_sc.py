import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def take_sc(performance, path):
    s = pd.Series(performance)
    rolling_mean = s.rolling(window=50).mean()

    plt.plot(np.arange(len(performance)), performance)
    plt.plot(np.arange(len(performance)), rolling_mean)
    plt.savefig(path)
    plt.clf()
