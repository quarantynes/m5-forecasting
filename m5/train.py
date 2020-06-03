from tqdm import tqdm
import tensorflow as tf

# Env setup
import m5.setup

# Training parameters
from m5.params import (
    batch_size,
    logdir,
    nb_epochs,
    training_days,
    validation_range,
    submission_range,
    nb_items,
)

# Model definition
from tensorflow.keras import layers


class StModel(tf.keras.models.Model):
    """
    This model predicts the unit sale for a given product_id, in a given day.
    There are 30490 different product_id while the time span in this dataset
    cover almost 2000 days.
    When submitting the prediction, this model should be evaluated 28 times for
    each product_id, one for each day in the submission range.
    The inputs of this model are:
    - product category (categorical)
    - product department (categorical)
    - week of the year (int)
    - day of the week (int)
    - snap in product_id state (boolean)
    """
    def __init__(self, dnn_units, **kwargs):
        super().__init__(**kwargs)
        self.DNN = tf.keras.Sequential(
            [
                layers.Dense(units=dnn_units,
                             activation=tf.keras.activations.relu),
                layers.Dense(units=dnn_units // 2,
                             activation=tf.keras.activations.relu),
                layers.Dense(units=dnn_units // 4,
                             activation=tf.keras.activations.relu),
                layers.Dense(1),
            ],
            name="DNN",
        ),

    def call(self, inputs, training=None, mask=None):
        output = self.DNN(inputs)
        return output


model = StModel(
    dnn_units=32,
    name="StModel",
)

# Data loading
from m5.data_loading import batch_generator
from m5.feature import (item_category, item_dept, item_state, reduced_calendar,
                        item_state_as_pandas)


def batch_generator(mode, batch_size):
    """ This generator returns either random samples from the training set or
    ordered samples from the validation dataset.
    If mode is:
      - 'train', it generates random samples from the training set.
      - 'validation', it generates ordered samples from the validation set.
      - 'submission', it generates ordered samples from the submission set.
    """
    assert mode in ['train', 'submission', 'validation']
    if mode == 'train':
        # nb_sample_days = 128
        # nb_sample_items = 512
        # batch_size = nb_sample_days * nb_sample_items
        from numpy.random import randint
        days_index = randint(training_days, size=batch_size)
        items_index = randint(nb_items, size=batch_size)

        feat_item_category, _ = item_category()
        feat_item_category = feat_item_category.take(items_index)

        feat_item_dept, _ = item_dept()
        feat_item_dept = feat_item_dept.take(items_index)

        feat_calendar = reduced_calendar()
        feat_calendar = feat_calendar.take(days_index)

        snap_columns = [
            col for col in feat_calendar.columns if col.startswith('snap_')
        ]
        feat_snap = feat_calendar.loc[snap_columns+['d']]
        state_df = item_state_as_pandas()
        feat_snap = feat_snap.set_index('d').lookup(
            state_df.d,
            state_df.state_id
        )
        # TODO: set column names for lookup

# IO:
def write_output(H, dir="evaluation"):
    import pandas as pd
    df = pd.DataFrame(H.numpy(), columns=[f"F{i}" for i in range(1, 1 + 28)])
    from m5.feature import item_id
    df.insert(0, "id", item_id())
    step = tf.summary.experimental.get_step()
    df.to_csv(f"data/{dir}/output_on_step{step}.csv", index=False)


# Training
def increment_step():
    step = tf.summary.experimental.get_step()
    if step is None:
        tf.summary.experimental.set_step(0)
    else:
        tf.summary.experimental.set_step(step + 1)


def train_batch(model, X, Y, w):
    with tf.GradientTape() as tape:
        H = model(X)
        loss = tf.losses.mean_squared_error(Y, H)
        loss = loss**0.5
        loss = loss * w
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss


def evaluate(model, eval_slice=slice(-28, None)):
    loss_list = []
    Hlist = []
    for (X, Y, w) in (batch_generator(eval_data=True, batch_size=10)):
        H = model(X)
        H = H[:, eval_slice]
        Y = Y[:, eval_slice]
        loss_i = tf.losses.mean_squared_error(Y, H)
        loss_i = loss_i**0.5
        loss_i = loss_i * w
        loss_list.append(loss_i)
        Hlist.append(H)
    loss = tf.concat(loss_list, axis=0)
    H = tf.concat(Hlist, axis=0)
    return H, loss


def evaluate_now(evaluation_period=evaluation_period) -> bool:
    step = tf.summary.experimental.get_step()
    return (step % evaluation_period) == 0


# Training Loop
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

with tf.summary.create_file_writer(logdir).as_default():
    for epoch in range(nb_epochs):
        print(f"Epoch: {epoch}")

        for (X, Y, w) in tqdm(batch_generator(batch_size=batch_size)):
            increment_step()
            batch_loss = train_batch(model, X, Y, w)
            tf.summary.scalar(f"{model.name}_batch_loss",
                              tf.reduce_mean(batch_loss))

            if evaluate_now():
                H, loss = evaluate(model)
                tf.summary.scalar(f"{model.name}_eval_loss",
                                  tf.reduce_mean(loss))
                write_output(H, "evaluation")

# Save Model
# TODO: save model here and implement prediction/submission
X, Y, w = batch_generator(eval_data=True, batch_size=batch_size)
model.save(f"../data/saved_models/{model.name}.tf")
