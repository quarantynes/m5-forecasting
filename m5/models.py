import tensorflow as tf
from tensorflow_core.python.keras.api._v2.keras import layers



class ModelV1(tf.keras.models.Model):
    def __init__(self, gru_units, dnn_units, nonCuda=False, **kwargs):
        super().__init__(**kwargs)
        if nonCuda:
            self.shared_layer = layers.RNN(
                layers.GRUCell(gru_units),
                return_sequences=True,
                name="sharedGru"
            )
        else:
            self.shared_layer = layers.GRU(
                gru_units,
                return_sequences=True,
                # stateful=True,
                name="shared_GRU",
            )
        self.DNN = layers.TimeDistributed(
            tf.keras.Sequential(
                [
                    # layers.Dense(units=dnn_units, activation=tf.keras.activations.relu),
                    # layers.Dense(units=dnn_units//2, activation=tf.keras.activations.relu),
                    # layers.Dense(units=dnn_units//4, activation=tf.keras.activations.relu),
                    layers.Dense(1),
                ],
            ),
            name="shared_DNN",
        )

    def call(self, inputs, training=None, mask=None):
        gru_output = self.shared_layer(inputs)
        output = self.DNN(gru_output)
        output = tf.squeeze(output)
        return output