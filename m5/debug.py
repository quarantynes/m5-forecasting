import ipdb

# import trepan
import matplotlib

matplotlib.use("qt5agg")

import matplotlib.pyplot as plt

plt.ion()
import seaborn as sns

import pandas as pd


def hist_layer(layer):
    data = layer.weights[0].numpy().squeeze()
    return pd.Series(data).hist(bins=100)
