import numpy as np

from m5 import feature
from m5.preprocess import memory
from m5.params import window, nb_items, batch_size, days


def shift_trunc_data(X):
    return X[:, 0:days - window], X[:, window:days]


@memory.cache()
def compose_data():
    units_sales = (feature.unit_sales_per_item_over_time()).astype(np.float32)
    assert units_sales.shape == (
        30490,
        1913,
    )
    prices = (feature.prices_per_item_over_time())
    weights = feature.sales_weight() / feature.item_weight()
    units_sales_feature, units_sales_target = shift_trunc_data(units_sales)
    prices, _ = shift_trunc_data(prices)

    assert units_sales_feature.shape == (
        30490,
        1913 - window,
    )
    assert prices.shape == (
        30490,
        1913 - window,
    ), prices.shape

    X = np.dstack([units_sales_feature, prices])
    Y = units_sales_target
    return X, Y, weights


def split_data(split_day=-28, sequence_training=True):
    X, Y, w = compose_data()
    train_idx = slice(0, split_day)
    if sequence_training:
        # for sequence training, we need the whole dataset to evaluation 
        eval_idx = slice(None, None)
    else:
        eval_idx = slice(split_day, None)

    return X[:, train_idx], Y[:, train_idx], X[:, eval_idx], Y[:, eval_idx], w


def batch_generator(eval_data=False, batch_size=batch_size):
    X, Y, Xeval, Yeval, weights = split_data()
    if eval_data:
        X, Y = Xeval, Yeval
        batch_size = nb_items  # single batch
    for i in range(nb_items // batch_size):
        samples = range(i * batch_size, min(
            (i + 1) * batch_size,
            nb_items,
        ))
        yield X[samples], Y[samples], weights[samples]

