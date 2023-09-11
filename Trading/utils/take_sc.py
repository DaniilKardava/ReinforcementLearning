import matplotlib.pyplot as plt
import numpy as np


def take_sc(performance, beta, scale, path):
    ema_corrected = np.zeros_like(performance)
    m = 0  # Initialize the moving average
    for t in range(len(performance)):
        m = beta * m + (1 - beta) * performance[t]
        # Compute bias-corrected EMA
        ema_corrected[t] = m / (1 - beta ** (t + 1))

    if scale == "symlog":
        plt.yscale(scale, linthresh = .0001)
    else:
        plt.yscale(scale)
    plt.plot(np.arange(len(ema_corrected)), ema_corrected)
    plt.savefig(path)
    plt.clf()
