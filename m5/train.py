from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers
# Env setup

# Training parameters
from m5.params import (
    batch_size,
    logdir,
    nb_epochs,
    training_days,
    evaluation_range,
    submission_range,
    nb_items,
    steps_per_epoch,
)

# Data loading
from m5.feature import (item_category, item_dept, item_state, reduced_calendar,
                        unit_sales_per_item_over_time, item_weight)

import pandas as pd


def make_batch(items_index, days_index):
    """Computes one batch for given items and days"""
    feat_item_category, _ = item_category()
    feat_item_category = feat_item_category.take(items_index)

    feat_item_dept, _ = item_dept()
    feat_item_dept = feat_item_dept.take(items_index)

    feat_calendar = reduced_calendar()
    feat_calendar = feat_calendar.take(days_index)

    # load item_state as pandas series
    state_series = pd.Categorical.from_codes(*item_state())

    # split feat_calendar into feat_calendar and snap_df
    # then, rename snap_df columns to match the values in state_series
    snap_df = feat_calendar[['snap_CA', 'snap_TX', 'snap_WI']]
    feat_calendar = feat_calendar.drop(
        columns=['snap_CA', 'snap_TX', 'snap_WI'])
    snap_df = snap_df.rename(columns={
        'snap_CA': 'CA',
        'snap_TX': 'TX',
        'snap_WI': 'WI'
    })

    # remove duplicated items in snap_df
    # note that drop_duplicates applies only to columns
    snap_df = snap_df.reset_index().drop_duplicates().set_index('index')

    # bi-dimensional lookup on snap_df to get snap information for
    # given items and given days
    feat_snap = snap_df.lookup(days_index, state_series[items_index])

    features = dict(
        category=feat_item_category,
        dept=feat_item_dept,
        weekday=feat_calendar.wday.values,
        month=feat_calendar.month.values,
        year=feat_calendar.year.values,
        snap=feat_snap,
    )

    try:
        target = unit_sales_per_item_over_time()[items_index, days_index]
    except IndexError:
        target = None

    weight = item_weight()[items_index]
    return features, target, weight


def batch_generator(mode, batch_size):
    """ This generator returns either random samples from the training set or
    ordered samples from the validation dataset. Each batch is a tuple (x,y,w)
    If mode is:
      - 'train', it generates random samples from the training set.
      - 'evaluation', it generates ordered samples from the validation set.
      - 'submission', it generates ordered samples from the submission set.
    """
    if mode == 'train':
        while True:
            from numpy.random import randint
            days_index = randint(training_days, size=batch_size)
            items_index = randint(nb_items, size=batch_size)
            yield make_batch(items_index, days_index)

    elif mode == 'evaluation':
        for day_i in range(*evaluation_range):
            days_index = [day_i] * nb_items
            items_index = list(range(nb_items))
            yield make_batch(items_index, days_index)

    elif mode == 'submission':
        for day_i in range(*submission_range):
            days_index = [day_i] * nb_items
            items_index = list(range(nb_items))
            yield make_batch(items_index, days_index)

    else:
        raise ValueError(
            f"mode({mode}) should be train, evaluation or submission")


# Model definition



# TODO: make model for new batch loader
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
        self.dnn_units = dnn_units
        self.DNN = tf.keras.Sequential(
            [
                # layers.Dense(units=dnn_units,
                #              activation=tf.keras.activations.relu),
                # layers.Dense(units=dnn_units // 2,
                #              activation=tf.keras.activations.relu),
                # layers.Dense(units=dnn_units // 4,
                #              activation=tf.keras.activations.relu),
                layers.Dense(1),
            ],
            name="DNN",
        )
        # self.mydense = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        """
        In this model, inputs is a dict. The available features are those returned by make_batch()
        """
        L = layers.Embedding(7,
                             self.dnn_units,
                             input_length=1,
                             input_shape=[None, 1])(inputs['category'])
        output = self.DNN(L)
        return output


print("building model")
model = StModel(
    dnn_units=32,
    name="StModel",
)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
print("testing model with one single batch")
x,y,w = next(batch_generator(mode="train",batch_size=128))
model(x)
model.summary()
print("model summary may appear incomplete if using subclassed model definition.")


# Training


def increment_step():
    step = tf.summary.experimental.get_step()
    if step is None:
        tf.summary.experimental.set_step(1)
    else:
        tf.summary.experimental.set_step(step + 1)
    return tf.summary.experimental.get_step()


def train_batch(model, X, Y, w):
    with tf.GradientTape() as tape:
        H = model(X)
        loss = tf.losses.mean_squared_error(Y, H)
        loss = loss**0.5
        loss = loss * w
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss



def evaluate_now(evaluation_period=100) -> bool:
    """
    This method should be used to set evaluation independently of the epoch. It
    returns True if the current iteration is a multiple of evaluation_period.
    """
    step = tf.summary.experimental.get_step()
    return (step % evaluation_period) == 0


def evaluate(model):
    """ Computes the prediction H and the loss for the whole evaluation range. H
    is a TF array in which the number of rows corresponds to the number of
    items, whereas the number of columns corresponds to the number of days in
    the evaluation range. """
    print("Entering in evaluate function.")
    loss_list = []
    Hlist = []
    for (X, Y, w) in (batch_generator(mode='evaluation', batch_size=None)):
        H = model(X)
        loss_i = tf.losses.mean_squared_error(Y, H)
        loss_i = loss_i**0.5
        loss_i = loss_i * w
        # next, insert dummy dimension to prepare concatenation
        loss_i = tf.expand_dims(loss_i,axis=-1)
        loss_list.append(loss_i)
        Hlist.append(H)
    loss = tf.concat(loss_list, axis=1)
    H = tf.concat(Hlist, axis=1)
    return H, loss


def submit(model):
    #TODO: finishe submit()
    raise NotImplementedError
    Hlist = []
    for (X, _, w) in (batch_generator(mode='submission', batch_size=None)):
        H = model(X)
        Hlist.append(H)
    H = tf.concat(Hlist, axis=0)
    return H


def write_output(H, dir="evaluation"):
    """ Writes the TF array H to csv using a pandas DataFrame
    """
    H = H.numpy()
    assert  H.shape == (30490,28)
    # using pandas fo IO
    import pandas as pd
    # create columns
    df = pd.DataFrame(H, columns=[f"F{i}" for i in range(1, 1 + 28)])
    # create index
    from m5.feature import item_id
    df.insert(0, "id", item_id())
    # write to file
    step = tf.summary.experimental.get_step()
    df.to_csv(f"data/{dir}/output_on_step{step}.csv", index=False)


def train_loop():
    with tf.summary.create_file_writer(logdir).as_default():
        for epoch in range(nb_epochs):
            print(f"Epoch: {epoch}")

            for (X, Y, w) in tqdm(
                    batch_generator(mode='train', batch_size=batch_size)):
                if increment_step() % steps_per_epoch == 0:
                    # need break here because the generator is infinite
                    break

                batch_loss = train_batch(model, X, Y, w)
                tf.summary.scalar(f"{model.name}_batch_loss",
                                  tf.reduce_mean(batch_loss))

            # Evaluate: record on loss on tensorboard and write predictions for
            # evaluation period in csv format
            H, loss = evaluate(model) tf.summary.scalar(f"{model.name}_eval_loss",
            tf.reduce_mean(loss)) write_output(H, "evaluation") print("finish
            train_loop") return


if __name__ == 'main':
    # Training Loop
    train_loop()

    # Save Model
    # TODO: save model here and implement prediction/submission
    X, Y, w = batch_generator(eval_data=True, batch_size=batch_size)
    model.save(f"../data/saved_models/{model.name}.tf")
