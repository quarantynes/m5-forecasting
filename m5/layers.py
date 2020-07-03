import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


class Price(layers.Layer):
    """This layer is an input layer. It loads data at init and receives indexes as
    inputs at call method"""

    def __init__(self, prices_array, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prices = tf.convert_to_tensor(prices_array)
        self.mean_price = tf.reduce_mean(self.prices, axis=1)

        # self.relative_price = self.prices - tf.stack(1969 * [self.mean_price], axis=1)
        self.relative_price = self.prices / tf.stack(1969 * [self.mean_price], axis=1)

        tf.debugging.assert_shapes(
            [
                (self.prices, (30490, 1969)),
                (self.mean_price, (30490,)),
                (self.relative_price, (30490, 1969)),
            ]
        )

    def call(self, inputs, training=None, mask=None):
        days_index = inputs["days_index"]
        items_index = inputs["items_index"]
        bi_dim_index = tf.stack([items_index, days_index], axis=-1)
        prices = tf.gather_nd(self.prices, bi_dim_index)
        mean_price = tf.gather(self.mean_price, items_index)
        relative_price = tf.gather_nd(self.relative_price, bi_dim_index)

        return prices, mean_price, relative_price


class Events(layers.Layer):
    """This layer is an input layer. It loads data at init and receives indexes as
    inputs at call method"""

    def __init__(self, events, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.events = tf.convert_to_tensor(events)
        self.events = tf.cast(self.events, tf.float32)
        tf.debugging.assert_shapes(
            [(self.events, (1969, 31)),]
        )
        self.dense = layers.Dense(5)

    def call(self, inputs, training=None, mask=None):
        days_index = inputs["days_index"]
        events = tf.gather(self.events, days_index)
        events = self.dense(events)
        tf.debugging.assert_shapes(
            [(events, ("d", 5)), (days_index, ("d")),]
        )
        return events
